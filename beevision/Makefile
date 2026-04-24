.PHONY: setup env download preprocess shards test sanity lint format clean \
        ingest sbatch-ingest sbatch-shards \
        train-seg sbatch-train-seg train-mite infer ablation

PY := python
CONFIG := configs/data.yaml
# Which ingest source to run; override on CLI: make ingest SRC=deepbee_seg
SRC ?= varroa

setup:
	$(PY) -m pip install -e ".[dev]"

env:
	conda env create -f environment.yml || conda env update -f environment.yml --prune

download:
	$(PY) -m beevision.data.download --config $(CONFIG)

preprocess:
	bash scripts/run_preprocess.sh $(CONFIG)

# Run one source ingest interactively (foreground).
#   make ingest SRC=deepbee_seg
ingest:
	$(PY) -m beevision.data.ingest_$(SRC) --config $(CONFIG)

# Submit one source ingest as a SLURM batch job.
#   make sbatch-ingest SRC=deepbee_seg
# sbatch prints the job id; logs stream to interim/_logs/slurm-*-<jobid>.{out,err}.
sbatch-ingest:
	sbatch --job-name=beevision-$(SRC) scripts/slurm_ingest.sbatch $(SRC)

# Submit the WebDataset shard writer as a SLURM batch job.
#   make sbatch-shards
#   make sbatch-shards ARGS="--sources varroa beeimage"
sbatch-shards:
	sbatch scripts/slurm_shards.sbatch $(ARGS)

shards:
	$(PY) -m beevision.data.shard_writer --config $(CONFIG)

sanity:
	$(PY) scripts/sanity_check.py --config $(CONFIG)

test:
	pytest -q

train-seg:
	bash scripts/train_seg.sh

# Submit segmentation training as a SLURM GPU job.
#   make sbatch-train-seg
#   make sbatch-train-seg TRAIN_CONFIG=configs/custom.yaml
sbatch-train-seg:
	sbatch scripts/slurm_train_seg.sbatch $(TRAIN_CONFIG)

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
