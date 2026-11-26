"""Evaluate the navigation and recognition projects independently."""

import math
from pathlib import Path

from .io import FormatError, digest, number, read_json, write_json
from .projects import ProjectSession, check_projects
from .scoring import evaluate, score_video, validate_reference, validate_video_reference
from .simulation import recognize_videos, robot_attempt

CHALLENGE = Path("challenge")
VIDEOS = Path("datasets/videos")
DEFAULT_SEED = 20261126


def make_output(output, *input_dirs):
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(path).resolve()) for path in input_dirs):
        raise FormatError("结果目录请放在项目、赛道和数据目录之外")
    output.mkdir(parents=True, exist_ok=False)
    return output


def run_navigation(submission, output, route_id=None, max_seconds=None, seed=DEFAULT_SEED):
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise FormatError("试跑时长必须大于 0")
    data, files = check_projects(submission, "navigation")
    reference_path = CHALLENGE / "reference.json"
    reference = validate_reference(read_json(reference_path))
    routes = [route for route in reference["routes"] if route_id is None or route_id == route["id"]]
    if not routes:
        raise FormatError("没有这张地图")
    output = make_output(output, files["navigation"], CHALLENGE)
    run = {
        "reference_id": reference["reference_id"],
        "team": data["team"],
        "routes": [],
        "disturbance": [],
        "friction": [],
        "seed": seed,
        "sensors": data["navigation"]["sensors"],
        "model_errors": [],
    }
    write_json(output / "reference.json", reference)
    nominal = reference["nominal_friction"]
    attempts = [("normal", 0, nominal, "routes")]
    attempts += [
        (f"force_{force}", force, nominal, "disturbance") for force in reference["force_levels_N"]
    ]
    attempts += [(f"friction_{mu}", 0, mu, "friction") for mu in reference["friction_levels"]]
    for route in routes:
        for name, force, friction, group in attempts:
            log = output / "logs" / f"{route['id']}_{name}.log"
            try:
                with ProjectSession(
                    files["navigation"], data["navigation"], "navigation", log
                ) as policy:
                    item = robot_attempt(
                        policy,
                        CHALLENGE,
                        reference,
                        route,
                        output / route["id"] / name,
                        force,
                        max_seconds,
                        seed,
                        friction=friction,
                    )
                item.pop("result")
                for key in ("trace", "force_record"):
                    item[key] = str(Path(item[key]).relative_to(output))
                if group == "friction":
                    item["friction"] = friction
                run[group].append(item)
            except (FormatError, OSError) as exc:
                run["model_errors"].append(
                    {"route_id": route["id"], "attempt": name, "error": str(exc)}
                )
            write_json(output / "run.json", run)
    result = evaluate(run, reference, output)
    full_duration = max_seconds is None or max_seconds >= reference["rules"]["timeout_s"]
    report = {
        "task": "navigation",
        "team": data["team"],
        "score": result["navigation_control_score"],
        "maximum": 50,
        "complete": route_id is None and full_duration and not run["model_errors"],
        "scores": {key: result["scores"][key] for key in ("route", "disturbance")},
        "details": result["details"],
        "sensors": run["sensors"],
        "seed": seed,
        "model_errors": run["model_errors"],
        "routes": [route["id"] for route in routes],
        "time_limit_s": min(
            max_seconds or reference["rules"]["timeout_s"], reference["rules"]["timeout_s"]
        ),
        "input_sha256": {"reference": digest(reference_path)},
    }
    write_json(output / "report.json", report)
    return report


def run_recognition(submission, output, video_id=None):
    data, files = check_projects(submission, "recognition")
    truth_path = VIDEOS / "train/reference.json"
    manifest_path = VIDEOS / "manifest.json"
    truth = validate_video_reference(read_json(truth_path))
    manifest = read_json(manifest_path)
    videos = [item for item in manifest["videos"] if item["id"].startswith("train")]
    ids = {item["id"] for item in videos}
    if len(videos) != 8 or len(ids) != 8 or ids != {event["video_id"] for event in truth["events"]}:
        raise FormatError("八段练习视频需要与答案一一对应")
    for uid in ids:
        if sum(event["video_id"] == uid for event in truth["events"]) != 12:
            raise FormatError(f"{uid} 需要 12 个标牌答案")
    if video_id is not None:
        if video_id not in ids:
            raise FormatError("没有这段练习视频")
        videos = [item for item in videos if item["id"] == video_id]
        truth = dict(
            truth, events=[event for event in truth["events"] if event["video_id"] == video_id]
        )
    output = make_output(output, files["recognition"], VIDEOS)
    predictions = output / "video_predictions.json"
    with ProjectSession(
        files["recognition"], data["recognition"], "recognition", output / "logs/recognition.log"
    ) as reader:
        recognize_videos(reader, VIDEOS, videos, predictions)
    scores = score_video(read_json(predictions), truth, include_answers=True)
    report = {
        "task": "recognition",
        "team": data["team"],
        "score": scores["score"],
        "maximum": 20,
        "complete": video_id is None,
        "videos": [item["id"] for item in videos],
        "scores": {"recognition": scores},
        "input_sha256": {
            "video_manifest": digest(manifest_path),
            "video_reference": digest(truth_path),
        },
    }
    write_json(output / "report.json", report)
    return report


def combine(navigation, recognition):
    expected = {
        "navigation": (50, {"reference": CHALLENGE / "reference.json"}),
        "recognition": (
            20,
            {
                "video_manifest": VIDEOS / "manifest.json",
                "video_reference": VIDEOS / "train/reference.json",
            },
        ),
    }
    for kind, report in [("navigation", navigation), ("recognition", recognition)]:
        maximum, files = expected[kind]
        if (
            not isinstance(report, dict)
            or report.get("task") != kind
            or report.get("complete") is not True
        ):
            raise FormatError(f"请使用完整 {kind} 自评的 report.json")
        if number(report.get("score"), kind, 0) > maximum or report.get("maximum") != maximum:
            raise FormatError(f"{kind} 分值超出范围")
        if report.get("input_sha256") != {key: digest(path) for key, path in files.items()}:
            raise FormatError(f"{kind} 结果与当前公开自评数据不一致")
    if not navigation.get("team") or navigation["team"] != recognition.get("team"):
        raise FormatError("两份成绩的队伍名称不一致")
    return {
        "team": navigation["team"],
        "scores": {**navigation["scores"], **recognition["scores"]},
        "navigation_control_score": navigation["score"],
        "recognition_score": recognition["score"],
        "automated_score": navigation["score"] + recognition["score"],
        "automated_maximum": 70,
        "pending_manual": {"report": 30},
    }


def run_projects(submission, output, seed=DEFAULT_SEED):
    _, files = check_projects(submission, require_report=False)
    output = make_output(output, files["navigation"], files["recognition"], CHALLENGE, VIDEOS)
    navigation = run_navigation(submission, output / "navigation", seed=seed)
    recognition = run_recognition(submission, output / "recognition")
    result = combine(navigation, recognition)
    write_json(output / "report.json", result)
    return result
