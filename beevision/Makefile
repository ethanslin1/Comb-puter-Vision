.PHONY: setup env download preprocess shards test sanity lint format clean \
        train-seg train-mite infer ablation

PY := python
CONFIG := configs/data.yaml

setup:
	$(PY) -m pip install -e ".[dev]"

env:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

download:
	$(PY) -m beevision.data.download --config $(CONFIG)

preprocess:
	bash scripts/run_preprocess.sh $(CONFIG)

shards:
	$(PY) -m beevision.data.shard_writer --config $(CONFIG)

sanity:
	$(PY) scripts/sanity_check.py --config $(CONFIG)

test:
	pytest -q

train-seg:
	bash scripts/train_seg.sh

train-mite:
	bash scripts/train_mite.sh

infer:
	bash scripts/run_inference.sh $(IMAGE)

ablation:
	bash scripts/run_ablation.sh

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
