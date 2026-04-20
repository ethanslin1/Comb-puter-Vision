.PHONY: setup data train-seg train-mite infer test lint clean

PY := python

setup:
	$(PY) -m pip install -e ".[dev]"

data:
	bash scripts/download_data.sh
	bash scripts/preprocess_all.sh

train-seg:
	bash scripts/train_seg.sh

train-mite:
	bash scripts/train_mite.sh

infer:
	bash scripts/run_inference.sh $(IMAGE)

ablation:
	bash scripts/run_ablation.sh

test:
	pytest tests -v

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
