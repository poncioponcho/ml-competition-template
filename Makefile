# 竞赛项目 — 一键重跑（赛题常量全部来自 configs/default.yaml）
#
# 所有目标都幂等，常量全部来自 configs/default.yaml。
# PYTHON 可覆盖：make test PYTHON=/path/to/python

# PYTHON 可覆盖：make test PYTHON=/path/to/python
PYTHON ?= python3
CONFIG ?= configs/default.yaml
export PYTHONPATH := src
export BERRY_CONFIG := $(CONFIG)

.DEFAULT_GOAL := help
.PHONY: help test preflight data split train predict submission empty-submission verify evaluate clean clean-all

help:  ## 显示本帮助
	@grep -E '^[a-zA-Z_-]+:.*## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

test:  ## 运行全部测试（无 GPU、无网络）
	$(PYTHON) -m pytest tests/ -v

preflight:  ## 上传前身份预检：确认数据属于本赛题，防止交错比赛
	$(PYTHON) -m submit.identity --split testB

verify-pipeline:  ## 过拟合探针：长训练前必跑，几分钟验证整条训练/推理链
	$(PYTHON) scripts/verify_training_pipeline.py

data:  ## 检查 COCO 数据并输出统计（含源图分组报告）
	$(PYTHON) -m data.coco

split:  ## 按源图分组做 K 折划分（断言同源图不跨折）
	$(PYTHON) -m data.split_by_source

train:  ## 训练实例分割模型（需 torch/torchvision）
	$(PYTHON) scripts/train_segmentation.py

predict:  ## 对 testB 推理，产出 outputs/predictions/raw_predictions.json
	$(PYTHON) scripts/predict_test.py --split testB

empty-submission:  ## 生成保底提交（空预测，格式合法，官方校验通过）
	$(PYTHON) scripts/build_submission.py --empty

submission:  ## 用真实预测生成提交（result.json + solution_commit + b_submission.zip）
	$(PYTHON) scripts/build_submission.py --predictions outputs/predictions/raw_predictions.json

verify:  ## 重新校验已有 b_submission.zip（不改动任何文件）
	$(PYTHON) -m submit.verify_submission outputs/submissions/b_submission.zip

evaluate:  ## 用冻结的官方评分脚本在本地折上评分
	$(PYTHON) scripts/evaluate_local.py

clean:  ## 清理生成物（保留数据与已提交材料）
	rm -rf outputs/reports outputs/predictions outputs/checkpoints .pytest_tmp .pytest_cache

clean-all: clean  ## 额外清理提交产物
	rm -rf outputs/submissions
