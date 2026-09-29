# solution/ — 可运行模型包（打包进 solution.zip）

赛后复核会**断网运行**这个包，并把复现结果与榜单成绩比对（通常要求指标绝对差 ≤ 0.005）。
所以它必须自包含：不联网、不依赖归档外的任何文件。

## 目录结构（按官方要求）

```text
solution.zip
├── model/                 # 模型权重与网络定义
├── config/                # 推理配置
├── src/                   # 推理源码
├── inference.py           # 统一推理入口（官方契约）
├── requirements.txt       # 依赖及版本
└── README.md              # 本文件
```

本模板只提供 `inference.py` 的骨架与依赖清单，`model/` 里的网络定义与权重按赛题自己实现。

## 推理入口契约

```bash
python inference.py \
  --input_dir test_images/ \
  --weights model/your_weights.pt \
  --output_path reproduced_result.json
```

- 读 `--input_dir` 下所有图片，每张图**恰好一条**记录（无检测写 `"instances": []`，
  **不能删记录**）
- 每条记录：`{"image_id": <完整文件名含扩展名>, "instances": [...]}`
- 每个实例恰好三个字段：`category_id`（提交编号，见 `competition.submit_classes`）、
  `score`、`segmentation`（掩膜按**原图分辨率**编码的 COCO compressed RLE）
- 顶层恰好 `{"version": "1.0", "results": [...]}`
- 省略 `--weights` 时输出格式合法的空预测 —— 这是"模型还没训好"的合法状态，
  官方校验器会通过、得分为 0

## 两条容易踩的坑

**1. 推理参数必须与提交时一致。** 分辨率、阈值、多尺度列表、集成权重顺序 ——
任何一项不同，复现分数就对不上。**把这些参数放在一处**（如 `model/model.py` 的常量），
并在本 README 写明"复现榜单成绩请使用默认值"。

**2. 掩膜必须先还原到原图分辨率再编码。** 训练时图像被缩放（省时间），推理后要把掩膜
按最近邻放大回原图尺寸再二值化、再 RLE 编码。顺序错了分数会明显偏低。

## 打包后不要重新压缩

`solution_commit.txt` 声明的是**原始 `solution.zip` 的完整字节**的 SHA-256。
重新压缩会改变时间戳与成员顺序 → SHA 变化 → 该次提交无效。打包后请保留原文件。
