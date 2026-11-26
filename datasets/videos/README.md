# 视频识别自评数据

`train/` 包含 8 段练习视频与 [96 个标牌的答案](train/reference.json)。每段视频 10 分钟，分辨率 640 × 360，每秒 5 帧。摄像机沿走廊移动，每段依次出现 6 个数字牌和 6 个温度屏，两类标牌交替出现。

每段有 4 个清晰标牌、4 个中等难度标牌、2 个困难标牌和 2 个无法辨认的标牌。查看[清晰](preview_clear.jpg)、[中等](preview_medium.jpg)、[运动模糊](preview_motion.jpg)、[遮挡](preview_partial.jpg)和[无法辨认](preview_unreadable.jpg)的示例。

识别项目逐帧返回标牌位置、读数和可信度，无法辨认时报告原因。视频参数与文件名见[清单](manifest.json)，接入方式见[项目运行接口](../../docs/model-interface.md)。
