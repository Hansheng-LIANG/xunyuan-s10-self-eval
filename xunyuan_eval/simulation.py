"""Simulate a route with a participant controller and read the practice videos."""

import math
import random
from itertools import pairwise
from pathlib import Path

from .io import FormatError, contained, digest, read_json, write_json


def robot_attempt(
    session,
    challenge_root,
    reference,
    route,
    output,
    force_N=0,
    max_seconds=None,
    seed=20261126,
    friction=None,
):
    import mujoco
    import numpy as np

    from .mujoco_recorder import MuJoCoSampler
    from .recorder import TraceRecorder
    from .scoring import score_route
    from .sensors import Lidar, contact_friction, set_ground_friction

    challenge_root = Path(challenge_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_xml_path(str(challenge_root / route["model"]))
    data = mujoco.MjData(model)
    model.vis.map.znear = 0.01 / model.stat.extent
    control = read_json(challenge_root / "robot/control.json")
    names = control["leg_joint_names"] + control["wheel_joint_names"]
    qadr = [int(model.jnt_qposadr[model.joint(name).id]) for name in names]
    vadr = [int(model.jnt_dofadr[model.joint(name).id]) for name in names]
    defaults = np.asarray(control["default_joint_pos"], dtype=np.float32)
    data.qpos[qadr] = defaults
    data.ctrl[:12] = defaults[:12]
    data.ctrl[12:] = 0
    for _ in range(500):
        mujoco.mj_step(model, data)
    if friction is not None:
        set_ground_friction(model, friction)
    data.time = 0.0
    mujoco.mj_forward(model, data)
    cells = {tuple(cell) for cell in route["driveable_cells"]}
    step = route["cell_size_m"]

    def outside(points):
        return any((math.floor(p[0] / step), math.floor(p[1] / step)) not in cells for p in points)

    sampler = MuJoCoSampler(
        model,
        root_body="base_link",
        terrain_geom_names=["floor"],
        allowed_terrain_body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"],
        boundary_check=outside,
    )
    sensors = session.spec["sensors"]
    renderer = mujoco.Renderer(model, 144, 256) if "front_rgb" in sensors else None
    lidar = Lidar(model) if "lidar_points" in sensors else None
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0
    task = read_json(challenge_root / route["task"])
    codes = {"left": 1, "straight": 2, "right": 3}
    instructions = np.array(
        [codes[t["turn"]] for t in task["outbound"]]
        + [4]
        + [codes[t["turn"]] for t in task["return"]],
        np.int64,
    )
    previous = np.zeros(16, np.float32)
    image = None
    body = model.body("base_link").id
    rows = []
    states = []
    pulses = []
    edge = 0
    points = route["path_xy"]
    lengths = [math.dist(a, b) for a, b in pairwise(points)]
    total = sum(lengths)
    done_distance = 0.0
    pulse_start = None
    pulse_vector = None
    next_pulse = 0
    frame = -1
    inference_count = 0
    termination = "timeout"
    reason = "timeout"
    next_status = 1.0
    base0 = data.xpos[body].copy()
    deadline = (
        min(reference["rules"]["timeout_s"], max_seconds)
        if max_seconds is not None
        else reference["rules"]["timeout_s"]
    )
    rng = random.Random(str(seed) + "_" + route["id"])
    jitter = reference["disturbance"].get("progress_jitter_fraction", 0.04)
    pulse_positions = [
        f + rng.uniform(-jitter, jitter) for f in reference["disturbance"]["progress_fractions"]
    ]

    def sample(legal=True):
        row = sampler.sample(data, action_valid=legal)
        row["position_m"] = [round(x, 6) for x in row["position_m"]]
        row["envelope_points_m"] = [[round(x, 6) for x in p] for p in row["envelope_points_m"]]
        row["force_world_N"] = data.xfrc_applied[body, :3].tolist()
        if friction is not None:
            row["ground_contact_friction"] = contact_friction(model, data)
        return row

    def summary(row, status):
        return {
            "time_s": round(row["time_s"], 6),
            "status": status,
            "relative_position_m": [round(float(x), 6) for x in data.xpos[body] - base0],
            "roll_rad": round(row["roll_rad"], 6),
            "pitch_rad": round(row["pitch_rad"], 6),
            "collision": bool(row["forbidden_contacts"]),
        }

    points_cloud = np.empty((0, 3), np.float32)
    sensor_time = 0.0
    try:
        session.reset()
        with TraceRecorder(output / "physics.jsonl") as recorder:
            row = sample()
            rows.append(row)
            recorder.append(row)
            initial_failure = (
                "collision"
                if row["forbidden_contacts"]
                else "out_of_bounds"
                if row["out_of_bounds"]
                else "fall"
                if max(abs(row["roll_rad"]), abs(row["pitch_rad"]))
                > reference["rules"]["fall_angle_rad"]
                else None
            )
            if (
                not initial_failure
                and math.dist(row["position_m"][:2], points[0])
                > reference["rules"]["start_radius_m"]
            ):
                initial_failure = "wrong_start"
            if initial_failure:
                reason = initial_failure
                termination = "safety_stop"
            states.append(summary(row, -1 if initial_failure else 0))
            while not initial_failure and data.time < deadline - 1e-9:
                if frame % 4 == 3 or frame < 0:
                    if inference_count % 5 == 0:
                        if renderer is not None:
                            renderer.update_scene(data, camera="front", scene_option=option)
                            image = renderer.render()
                        if lidar is not None:
                            points_cloud = lidar.scan(data)
                        sensor_time = float(data.time)
                    rotation = data.xmat[body].reshape(3, 3)
                    velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(
                        model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0
                    )
                    imu = np.r_[
                        rotation.T @ velocity[:3], rotation.T @ np.array([0, 0, -1])
                    ].astype(np.float32)
                    feed = {
                        "imu": imu,
                        "joint_position": data.qpos[qadr].astype(np.float32),
                        "joint_velocity": data.qvel[vadr].astype(np.float32),
                        "previous_action": previous,
                        "instructions": instructions,
                        "sensor_time_s": np.array([sensor_time]),
                        "time_s": np.array([float(data.time)]),
                    }
                    if renderer is not None:
                        feed["front_rgb"] = image
                    if lidar is not None:
                        feed["lidar_points"] = points_cloud
                    try:
                        action = session.action(feed)
                    except FormatError:
                        reason = "model_error"
                        termination = "crashed"
                        break
                    legal = np.isfinite(action).all() and np.max(np.abs(action)) <= 1.00001
                    if not legal:
                        # Keep a strictly increasing final timestamp.
                        mujoco.mj_step(model, data)
                        mujoco.mj_forward(model, data)
                        row = sample(False)
                        rows.append(row)
                        recorder.append(row)
                        reason = "illegal_action"
                        termination = "safety_stop"
                        break
                    previous = action
                    data.ctrl[:12] = defaults[:12] + action[:12] * np.asarray(control["leg_scale"])
                    data.ctrl[12:] = action[12:] * control["wheel_scale"]
                    inference_count += 1
                position = data.xpos[body, :2]
                if (
                    edge < len(points) - 1
                    and math.dist(position, points[edge + 1])
                    <= reference["rules"]["checkpoint_radius_m"]
                ):
                    done_distance += lengths[edge]
                    edge += 1
                progress = done_distance
                if edge < len(lengths):
                    v = np.array(points[edge + 1]) - points[edge]
                    progress += (
                        float(np.clip((position - np.array(points[edge])) @ v / (v @ v), 0, 1))
                        * lengths[edge]
                    )
                if (
                    force_N
                    and pulse_start is None
                    and next_pulse < 3
                    and progress / total >= pulse_positions[next_pulse]
                ):
                    pulse_start = float(data.time)
                    rotation = data.xmat[body].reshape(3, 3)
                    direction = rotation[:2, 1].copy()
                    direction /= np.linalg.norm(direction)
                    pulse_vector = np.r_[direction * rng.choice([-1, 1]) * force_N, 0.0]
                    pulses.append(
                        {
                            "start_s": pulse_start,
                            "force_world_N": pulse_vector.tolist(),
                            "duration_s": 0.0,
                        }
                    )
                    next_pulse += 1
                data.xfrc_applied[body] = 0.0
                if pulse_start is not None:
                    if (
                        data.time - pulse_start
                        < reference["disturbance"]["pulse_duration_s"] - 1e-9
                    ):
                        data.xfrc_applied[body, :3] = pulse_vector
                        pulses[-1]["duration_s"] += model.opt.timestep
                    else:
                        pulse_start = None
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                frame += 1
                row = sample()
                rows.append(row)
                recorder.append(row)
                failure = (
                    "collision"
                    if row["forbidden_contacts"]
                    else "out_of_bounds"
                    if row["out_of_bounds"]
                    else "fall"
                    if max(abs(row["roll_rad"]), abs(row["pitch_rad"]))
                    > reference["rules"]["fall_angle_rad"]
                    else None
                )
                reached = edge == len(points) - 1
                if data.time + 1e-9 >= next_status or failure or reached:
                    states.append(summary(row, -1 if failure else 1 if reached else 0))
                    next_status = math.floor(data.time + 1e-8) + 1
                if failure:
                    reason = failure
                    termination = "safety_stop"
                    break
                if reached:
                    reason = "completed"
                    termination = "completed"
                    break
    finally:
        if renderer is not None:
            renderer.close()
    final = summary(rows[-1], 1 if termination == "completed" else -1)
    if states and abs(states[-1]["time_s"] - final["time_s"]) < 1e-8:
        states[-1] = final
    else:
        states.append(final)
    write_json(
        output / "states.json", {"states": states, "termination": termination, "reason": reason}
    )
    write_json(output / "forces.json", {"force_N": force_N, "pulses": pulses})
    judged = score_route(rows, route, reference["rules"], termination)
    if force_N and (
        len(pulses) != 3
        or any(
            abs(p["duration_s"] - reference["disturbance"]["pulse_duration_s"]) > 1e-8
            for p in pulses
        )
    ):
        judged["passed"] = False
    judged["runtime_reason"] = reason
    write_json(output / "result.json", judged)
    return {
        "route_id": route["id"],
        "force_N": force_N,
        "termination": termination,
        "trace": str((output / "physics.jsonl").resolve()),
        "force_record": str((output / "forces.json").resolve()),
        "result": judged,
    }


def recognize_videos(session, videos_root, selected, output):
    import cv2

    from .projects import validate_detections

    if not selected:
        raise FormatError("没有找到指定的视频")
    detections = []
    for item in selected:
        path = contained(Path(videos_root) / "train", item["file"])
        if digest(path) != item["sha256"]:
            raise FormatError("视频文件校验值不一致")
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        index = 0
        try:
            if not cap.isOpened() or fps <= 0:
                raise FormatError("无法读取视频")
            session.reset()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
                try:
                    rows = validate_detections(session.read_frame(rgb, index / fps, item["id"]))
                except FormatError as exc:
                    raise FormatError(f"{item['id']} 第 {index + 1} 帧：{exc}") from exc
                detections.extend(r for r in rows if r["confidence"] >= 0.5)
                index += 1
        finally:
            cap.release()
        if index != round(item["duration_s"] * item["fps"]) or abs(fps - item["fps"]) > 1e-6:
            raise FormatError("视频帧数或帧率与清单不一致")
    write_json(output, {"detections": detections})
    return output
