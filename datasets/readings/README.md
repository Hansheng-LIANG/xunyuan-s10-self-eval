# 数字牌与温度屏图片

[images/](images/) 包含 6 张数字牌和 6 张温度屏的合成图片。[reference.json](reference.json) 给出文件名、标牌编号和读数。数字牌保留前导零、负号和小数点，温度单位为摄氏度。

使用自己的 OCR 项目读取图片后，可以运行以下命令检查逐张结果。预测文件格式见[静态图片使用说明](../../docs/image-datasets.md)。

```bash
python -m xunyuan_eval readings \
  --predictions my_predictions.json \
  --reference datasets/readings/reference.json \
  --output artifacts/panel_reading_report.json
```
