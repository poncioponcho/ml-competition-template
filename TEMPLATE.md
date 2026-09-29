# TEMPLATE.md — 换赛题时要改什么

从零开一个竞赛项目最贵的不是模型，是**提交链路**和**验证纪律**：提交格式错一次就少一次机会，
验证口径错一次就选错模型。这个仓库把这两块固化下来，换赛题时只需替换下面列出的东西。

> 顺序建议：先做第 1、2 节（身份与配置），再跑第 3 节的自检，最后才训练。

---

## 1. 必改：赛题身份（防交错比赛）

工作区里往往同时存在多个赛题，交错提交不可逆。三道防线都靠配置驱动：

| 改什么 | 在哪 |
|---|---|
| `competition.id` / `slug` / `platform` / `name` / `official_page` | `configs/default.yaml` |
| `competition.not_this_competition` | 同上，列出容易混淆的相邻赛题及其差异 |
| `competition.schedule` / `deadline` / `submit_limit` | 同上 |
| 身份预检的具体断言 | `src/submit/identity.py` |

`src/submit/identity.py` 是**唯一需要动代码**的身份文件。预检默认检查：
测试清单存在、图像数等于 `competition.n_test_images`、尺寸等于 `competition.image_size`、
每张图在磁盘上存在。**如果本赛题有更强的身份特征（例如"53 张原图 + 53 张翻转图"
这种配对规律），一定要加进去** —— 那是性价比最高的一条断言。

`build_submission.py` 在打包前会跑这些断言，任何一项失败就**拒绝构建**。

---

## 2. 必改：赛题契约

| 改什么 | 在哪 | 注意 |
|---|---|---|
| 提交类别编号 `submit_classes` / `submit_labels` | `configs/default.yaml` | 提交编号与训练 COCO 编号**经常不同**，写错会把两个类对调 |
| `gt_to_submit_category` | 同上 | 转换只在 `src/data/coco.py` 一处完成，别在别处再转一次 |
| 指标 `metric` / `metric_name` / `iou_thresholds` | 同上 | 只是记录；真正裁决的是官方脚本 |
| 数据规模 `n_train_images` / `n_test_images` / `image_size` | 同上 | 跑 `make data` 拿到实测值再回填 |
| 数据路径 `data.*` | 同上 | |
| 源图分组规则 `data.source_group` | 同上 | **本仓库最关键的一项**，见第 4 节 |
| 提交包结构 `submit.*` | 同上 | 文件名、字段顺序、体积上限 |
| `competition.data_version` | 同上 | 会写进 `solution_commit.txt` |

---

## 3. 必做：把官方脚本冻结进来

官方评分器与校验器**逐字节复制**进 `src/`，SHA-256 记到 `src/official_freeze.json`：

```bash
# 1) 从官方提交包解压（路径按官方包的目录结构）
cp <官方包>/official_evaluate.py      src/eval/official_oracle/
cp <官方包>/rle_validation.py         src/eval/official_oracle/     # 若官方提供
cp <官方包>/validate_b_submission.py  src/submit/official/
cp <官方包>/submission.py             src/submit/official/

# 2) 记录 SHA-256（可附来源，便于日后追溯）
python scripts/freeze_official.py --source data/raw/<官方包>.zip

# 3) 冻结测试必须绿
make test
```

`scripts/freeze_official.py` 会扫描那两个目录、写入 `src/official_freeze.json`。
manifest 为空时冻结测试会 **skip 并提示**，所以还没解压官方包也不会让 `make test` 变红。

**为什么必须冻结**：可编辑的评分脚本就是会被编辑的评分脚本，改完之后它产出的所有数字都不可信。
`tests/test_official_freeze.py` 会在文件被改动时红灯。缺脚本时它会 **skip 而非 fail**，
所以别人 clone 下来直接 `make test` 也是绿的。

**这两个目录不要提交到公开仓库** —— 是主办方的第三方代码。私有仓库另说。

---

## 4. 必查：本地验证口径（最容易自欺的一环）

**问题**：同一张源图往往有多个版本（原图 + 增强图，如 `sample.jpg` / `sample_aug1.jpg`）。
按图随机划分会把它们分到不同折，验证分数虚高，选模型全错。

**做法**：`src/data/split_by_source.py` 按源图分组做 K 折，并用**硬断言**挡住泄漏
（`assert_no_source_leakage` / `assert_full_coverage`），不是靠约定。

换赛题时要改的：`data.source_group.aug_suffix` 与 `strip_suffixes`。

**报告规则**：只有「按源图分组留出折 + 冻结官方脚本」得到的分数才算证据。
测试集无标注时，任何其他口径的数字都必须标成 `LEAKY`，不得用于决策。

### 配对验证纪律（血的教训）

**任何改动的结论必须来自 5 折配对比较。**

单折 140 张图的 mAP 噪声在 **±0.7 pp** 量级。真实案例：某次改动在 fold-4 上看着涨了
**+0.88 pp**（像一场明确胜利），补齐 5 折后均值只有 **+0.18 pp**、标准差 0.69 pp ——
与 0 无法区分。这个改动因此被否掉。

采纳门槛：**均值 ≥ 0.5 pp 且不能只有 1 折变好**。

```bash
# 逐折跑同一配置，再算均值/标准差/胜负比
for f in 0 1 2 3 4; do python scripts/evaluate_local.py --fold $f --out outputs/reports/fold${f}.json; done
```

---

## 5. 提交链路自检（每次改完都跑）

```bash
make test                # 全部测试，无 GPU 无网络
make preflight           # 身份预检：数据真的是本赛题的吗
make empty-submission    # 保底提交：格式合法、官方校验通过、得分为 0
make verify              # 只校验已有提交包，不改动任何文件
```

**保底提交先落地**：截止前永远有一次有效提交，是防止"最后一小时翻车"的最低成本保险。

### 提交包的两条硬约束

1. 根目录**恰好**是配置里列出的那几个普通文件 —— 不得有目录项、符号链接、加密文件、
   重复文件名。`pack_submission.py` 与 `verify_submission.py` 已硬拦截。
2. `solution_commit.txt` 恰好 5 个字段、UTF-8、无注释无空行；
   SHA-256 计算对象是**原始 `solution.zip` 的完整字节**。
   **打包后绝不重新压缩** —— 重压会改 SHA，声明立刻失效。

### 上传前 30 秒人工确认

平台 URL 里的 `type=` 等于 `competition.id`；页面标题等于 `competition.name`；
上传的是配置里 `submit.b_zip_name` 指定的那个文件。

### ⚠️ 平台通常取"最晚"的有效提交，不取最高分

所以**末位改动有下行风险**：交一版更差的，会把好成绩覆盖掉。
未经本地验证的改动不要交，宁可不用完次数。

---

## 6. 提交后：可复现性检查

赛后复核会**断网运行**你的 `solution.zip` 并比对分数。用 `scripts/repro_check.py` 提前验证：

```bash
python scripts/repro_check.py \
    --solution-zip outputs/submissions/solution.zip \
    --predictions outputs/predictions/raw_predictions.json \
    --manifest <config: data.test_manifest> \
    --num-images 12
```

它会解压归档、用**包内自己的** `inference.py` 重跑一小批图，再与提交的预测比对。

**判定口径：指标相关项严格、浮点项带容差。**
跨设备（训练在 MPS，复核在 CUDA/CPU）逐字节相等是不现实的目标：

- `category_id` 严格相等
- 掩膜按 **IoU ≥ 0.995**（边界像素翻转属设备噪声，形状不同才是 bug）
- `score` 容差 **1e-5**

包内推理的**分辨率/阈值等参数必须与提交时一致**，否则复现算不出同样分数 ——
把这些参数放在一处（如 `solution/model/model.py` 的常量），并在包内 README 写明。

---

## 7. 换赛题时的清理清单

- [ ] `configs/default.yaml` 所有 `REPLACE_ME` 已填
- [ ] `src/submit/identity.py` 的断言按本赛题改写（含最强身份特征）
- [ ] 官方脚本已冻结，`make test` 绿
- [ ] `data.source_group` 按本赛题命名规则填写
- [ ] 数据规模数字已用 `make data` 实测回填
- [ ] `solution/` 换成自己的模型与推理入口，分辨率参数写在一处
- [ ] `README.md` 改写（本模板的 README 是给模板本身用的）
- [ ] `.gitignore` 确认：数据与权重不入库，排除目录**加前导斜杠**、注释**单独成行**
- [ ] 大文件（权重、solution.zip）走 Release 附件或外部备份，不要硬塞进 git

---

## 8. 这个模板不包含什么

有意不包含，需要你按赛题自己写：

| 缺什么 | 为什么 |
|---|---|
| 具体模型与训练循环 | 模板里只有 `scripts/train_segmentation.py` 的实例分割实现；分类/检测要自己写 |
| `solution/model/model.py` 的推理实现 | 与模型强绑定，照 `solution/README.md` 的契约自己实现 |
| 官方评分/校验脚本 | 第三方代码，需从官方提交包解压 |
| 任何赛题数据与权重 | 不入库 |

`archive/` 目录也没有带过来 —— 那是我上一个赛题的归档代码，对新项目只有干扰。
