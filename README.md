# ml-competition-template

数据竞赛（Kaggle / 讯飞等）的**工程层模板**：把提交链路、身份预检、本地验证纪律固化下来，
换赛题时只替换赛题层，不重复踩坑。

> **换赛题请先读 [`TEMPLATE.md`](TEMPLATE.md)** —— 那里逐条列出要改什么、为什么。

## 它解决什么问题

从零开一个竞赛项目，最贵的不是模型，是这两块：

1. **提交链路**：提交格式错一次就少一次机会。根目录多一个文件、SHA 声明格式错、
   重压一次 `solution.zip` —— 任何一条都会让提交无效，而且往往在截止前才发现。
2. **验证纪律**：本地分数虚高会让你选错模型。同一张源图的多个版本跨折、按图随机划分、
   只看单折的涨跌，都是常见自欺。

本模板把这两块做成可执行的东西，而不是文档里的约定。

## 三个设计要点

**1. 常量只有一个事实源。** `configs/default.yaml` 集中所有决策常量，`src/**`、
`scripts/**`、`tests/**` 一律不得硬编码。

**2. 身份预检在构建前拦人。** 工作区常同时存在多个赛题，交错提交不可逆。
`src/submit/identity.py` 在打包前断言"数据真的是本赛题的数据"（图像数、尺寸、
命名规律、最强身份特征），任何一项失败就**拒绝构建**。

**3. 官方脚本逐字节冻结。** 评分器与校验器复制进 `src/`，SHA-256 记录在案，
测试会在文件被改动时红灯 —— 评分口径不可被悄悄改动。

## 目录

```text
configs/default.yaml          赛题常量唯一事实源（换赛题改这里）
src/common/                   配置加载、SHA-256、IO
src/data/
  coco.py                     COCO 读取 + 类别重编号（提交编号 vs 训练编号）
  split_by_source.py          按源图分组 K 折 + 泄漏硬断言  ← 最值钱的资产
src/eval/
  mask_map.py                 调用冻结的官方评分脚本
  official_oracle/            ⛔ 官方评分脚本（第三方代码，需自己放入）
src/submit/
  result_json.py              result.json 构造与逐字段校验
  solution_commit.py          5 字段 SHA 声明
  pack_submission.py          打包（根目录扁平约束）
  verify_submission.py        本地检查 + 调用官方校验器
  identity.py                 赛题身份预检（防交错比赛）
  official/                   ⛔ 官方校验器（第三方代码，需自己放入）
scripts/
  train_segmentation.py       训练（含续训学习率修复，见下）
  predict_test.py             推理（支持多尺度 TTA 融合）
  evaluate_local.py           留出折评分（走官方脚本）
  build_submission.py         打包 → 声明 → 构建 → 官方校验，一条链
  repro_check.py              可复现性检查（跨设备容差判定）
  threshold_scan.py           推理阈值扫描
  night_runner.py             带决策门槛的夜间串行任务队列
  verify_training_pipeline.py 过拟合探针：长训练前先验证整条链路
solution/                     可运行模型包（打包进 solution.zip）
tests/                        单元测试，无 GPU 无网络
TEMPLATE.md                   ★ 换赛题填空清单
```

## 快速开始

```bash
make test                # 全部测试，无 GPU 无网络
make help                # 列出所有目标
make preflight           # 身份预检
make empty-submission    # 保底提交（格式合法、官方校验通过、得分 0）
```

填好 `configs/default.yaml` 之后：

```bash
make data                # 数据统计（含源图分组报告）
make split               # 分组 K 折（断言同源图不跨折）
make train               # 训练，逐折保存 checkpoint
make evaluate            # 用冻结的官方脚本在留出折上评分
make predict             # 对测试集推理
make submission          # 打包 + 声明 + 校验，一条链
make verify              # 只校验已有提交包
```

## 内置的几条硬教训

这些都是实际踩过并付出代价的，所以写进了代码和测试：

| 教训 | 落地位置 |
|---|---|
| **任何改动的结论必须来自 5 折配对比较**（单折 mAP 噪声 ±0.7 pp，只看一折会误判） | `TEMPLATE.md` 第 4 节 |
| **"跑满轮数 + loss 正常" ≠ 在训练**：续训时 `optimizer.load_state_dict()` 会把 `lr` 一起恢复，cosine 跑满时 `lr=0` → 整个续训空转 | `make_scheduler()` + `tests/test_train_resume_scheduler.py` |
| **复现检查要按容差判定**：跨设备逐字节相等不现实，掩膜按 IoU、score 按容差 | `scripts/repro_check.py` + `tests/test_repro_check.py` |
| **多尺度/多成员融合时，收集张量的 append 必须与产生它们的层级同缩进**（单尺度下不可见的 bug） | `tests/test_multiscale_predict_image.py` |
| **平台取"最晚"的有效提交，不取最高分** → 末位改动有下行风险 | `TEMPLATE.md` 第 5 节 |
| **`.gitignore` 排除目录要加前导斜杠、注释要单独成行**（裸 `data/` 会误伤 `src/data/`；行尾注释会让规则失效） | `.gitignore` |

## 大文件怎么处理

权重（数百 MB/个）和 `solution.zip`（常 >1 GB）**放不进 git**：GitHub 对仓库内单文件有
100 MB 硬上限，private 仓库同样适用。

推荐 **GitHub Release 附件**：每个附件上限 2 GiB、单次 release 最多 1000 个、
无总量与带宽限制，且不进 git 历史。也可用外部硬盘或对象存储。

## 许可证

[MIT](LICENSE) © 2026 poncioponcho

**注意**：模板里的代码是 MIT；但用它搭出来的项目可能包含**不属于你**的内容 ——
比赛数据归主办方、官方评分/校验脚本是主办方提供的第三方代码。
这些不要放进公开仓库，也不要纳入你的许可证声明。

模板的 `.gitignore` 已经把数据与权重排除在外；官方脚本目录
（`src/eval/official_oracle/`、`src/submit/official/`）需要你自己判断是否提交。
