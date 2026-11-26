"""Command line interface for S10 participant self-evaluation."""

import argparse
import json
import sys
from pathlib import Path

from .io import FormatError, read_json, write_json
from .project_runtime import (
    DEFAULT_SEED,
    combine,
    run_navigation,
    run_projects,
    run_recognition,
)
from .projects import check_projects, package_projects, smoke_projects
from .reading_dataset import reading_reference
from .readings import evaluate_readings


def main(argv=None):
    parser = argparse.ArgumentParser(description="巡渊 S10 参赛队伍自评")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("navigation", "导航运控自评（50 分）"),
        ("recognition", "视频识别自评（20 分）"),
        ("run-submission", "运行两项完整自评（70 分）"),
        ("package-models", "将完整项目与 PDF 报告打包"),
        ("check-submission", "检查提交或单项项目接口"),
    ):
        cmd = commands.add_parser(name, help=description)
        cmd.add_argument("--submission", type=Path, required=True)
        if name != "check-submission":
            cmd.add_argument("--output", type=Path, required=True)
        if name in ("navigation", "run-submission"):
            cmd.add_argument("--seed", type=int, default=DEFAULT_SEED, help="推力扰动的随机种子")
        if name == "navigation":
            cmd.add_argument("--route", help="只跑指定地图，如 map01")
            cmd.add_argument("--max-seconds", type=float, help="每次尝试的仿真时长上限")
        if name == "recognition":
            cmd.add_argument("--video", help="只跑指定练习视频，如 train01")
        if name == "check-submission":
            cmd.add_argument("--task", choices=("navigation", "recognition"))
            cmd.add_argument("--smoke", action="store_true", help="启动程序并调用一次模型")

    cmd = commands.add_parser("combine", help="合并两个单项的完整成绩")
    cmd.add_argument("--navigation", type=Path, required=True)
    cmd.add_argument("--recognition", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)

    cmd = commands.add_parser("readings", help="对照静态图片答案检查读数")
    cmd.add_argument("--predictions", type=Path, required=True)
    cmd.add_argument("--reference", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    cmd = commands.add_parser("reading-reference", help="从静态图片标注生成分组答案")
    cmd.add_argument("--dataset", type=Path, required=True)
    cmd.add_argument("--split", choices=("train", "val", "test"), required=True)
    cmd.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "check-submission":
            if args.smoke:
                result = smoke_projects(args.submission, args.task)
            else:
                check_projects(args.submission, args.task)
                result = {"valid": True}
        elif args.command == "package-models":
            result = package_projects(args.submission, args.output)
        elif args.command == "combine":
            if args.output.exists():
                raise FormatError("请使用新的成绩文件名")
            result = combine(read_json(args.navigation), read_json(args.recognition))
            write_json(args.output, result)
            result = {"automated_score": result["automated_score"], "report": str(args.output)}
        elif args.command == "reading-reference":
            if args.output.resolve().is_relative_to(args.dataset.resolve()):
                raise FormatError("答案文件请保存到图片集目录之外")
            reference = reading_reference(args.dataset, args.split)
            write_json(args.output, reference)
            result = {"reference": str(args.output), "events": len(reference["events"])}
        elif args.command == "readings":
            if args.output.resolve() in (args.predictions.resolve(), args.reference.resolve()):
                raise FormatError("结果文件不能覆盖输入文件")
            result = evaluate_readings(read_json(args.predictions), read_json(args.reference))
            write_json(args.output, result)
        else:
            if args.command == "navigation":
                report = run_navigation(
                    args.submission, args.output, args.route, args.max_seconds, args.seed
                )
            elif args.command == "recognition":
                report = run_recognition(args.submission, args.output, args.video)
            else:
                report = run_projects(args.submission, args.output, args.seed)
            result = {
                key: report[key]
                for key in ("score", "maximum", "complete", "automated_score")
                if key in report
            }
            result["report"] = str(args.output / "report.json")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except ImportError as exc:
        print(
            '请先安装运行依赖：python -m pip install -e ".[runtime]"\n' + str(exc), file=sys.stderr
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"自评失败：{exc}", file=sys.stderr)
    return 2
