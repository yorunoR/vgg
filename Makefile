board:
	uv run tensorboard --logdir=. --port=5000

fmt:
	uv run ruff format

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix
