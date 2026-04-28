.PHONY: train infer test test-cov  lint lint-fix  format clean

train:
	uv run python main.py --epochs 1000 --save

infer:
	uv run python inference.py --features 2000 3

test:
	uv run python -m pytest

test-cov:
	uv run python -m pytest --cov=src

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

format:
	uv run ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache
