# 项目接入与打包

自评时，清单只需列出正在运行的项目。先生成导航或识别模板，替换其中的权重和推理逻辑。完整示例包含两个项目和一份示例 PDF。

```bash
python -m tools.make_project_example --task navigation --output artifacts/nav_team
python -m tools.make_project_example --task recognition --output artifacts/ocr_team
python -m tools.make_project_example --output my_team
```

每个项目目录包括以下内容：

| 文件 | 用途 |
|---|---|
| `run.py` | 示例进程入口。可换成其他语言或程序，并相应修改清单中的 `command`。 |
| `weights/` | 本项目的权重。格式由自己的推理代码决定。 |
| `requirements.txt` | 本项目的依赖清单。 |
| `AGENT.md` | 写明环境、安装方法、权重位置、启动命令、输入输出及检查方法。 |

将配置和预处理、后处理代码一并放进对应项目。清单中的项目目录及报告路径相对于 `submission.json`，权重与依赖路径相对于各自的项目目录。使用 `{python}` 启动时，测评器优先选用该项目的 `.venv/bin/python`，否则使用测评器当前的解释器。

[查看完整清单示例](../challenge/submission.example.json) · [项目通信接口](model-interface.md)

**检查一个项目。** 使用 `--task navigation` 或 `--task recognition`；加 `--smoke` 会启动该程序并发送一份输入。

```bash
python -m xunyuan_eval check-submission --task navigation \
  --submission artifacts/nav_team/submission.json --smoke
```

**打包参赛材料。** 把两个项目放在同一份清单中，填写 `team`，并用队伍自己的报告替换 `report.pdf`。下列命令检查两个项目及报告并生成 ZIP。

```bash
python -m xunyuan_eval package-models --submission my_team/submission.json --output artifacts/team.zip
```
