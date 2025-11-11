# Repository Guidelines

## Project Structure & Module Organization
- Source and notebooks live under `gandra/sessionXX/` (e.g., `gandra/session03/session03.py` mirrors the notebook). Root also contains some `session0X.ipynb` and exported `*.html` for lectures.
- Data: `_data/` (e.g., `italian_recipes_*`).
- Docs: `docs/` (environment with uv, pgvector how‑to).
- Prompts: `prompts/`.
- Infra: `docker-compose.yml` (Postgres + pgvector).
- Packaging: `pyproject.toml` + `uv.lock` (authoritative). `requirements.txt` is reference only.

## Build, Test, and Development Commands
- Create/refresh env and install deps: `uv sync`
- Launch notebooks: `uv run jupyter lab`
- Run demo script: `uv run python gandra/session03/session03.py`
- Start DB: `docker compose up -d pgvector` (exposes `localhost:5422`)
- Optional dev tools: `uv add --dev ruff pytest`
- Lint (if installed): `uv run ruff check .`
- Tests (if present): `uv run pytest -q`

## Coding Style & Naming Conventions
- Python 3.12, 4‑space indent, type hints preferred, module‑level docstrings.
- Names: modules `lower_snake.py`, functions `lower_snake`, classes `CapWords`, constants `UPPER_SNAKE`.
- Keep side effects under `if __name__ == "__main__":`. Favor small, pure functions.
- When adding notebooks, also add a mirrored `.py` with clearer structure/comments (see `session03.py`).

## Testing Guidelines
- Framework: pytest. Place tests under `tests/` as `test_*.py`.
- Mock external APIs (OpenAI) and guard integration tests with markers.
- DB integration tests require the container running (`docker compose up -d pgvector`). Use compose defaults or env vars for connection.
- Run with `uv run pytest -q`. Keep tests fast and deterministic.

## Commit & Pull Request Guidelines
- Commits: short, imperative, scoped prefixes optional (e.g., `docs: update pgvector notes`, `gandra: add session05 script`, `env: uv sync`). Avoid `wip` on main history.
- PRs: clear description, linked issues, steps to run (e.g., `uv sync`, `docker compose up -d pgvector`, `uv run python ...`), and screenshots/logs for visible changes. Update docs as needed.

## Security & Configuration Tips
- Do not commit secrets. `.env` is git‑ignored; set `OPENAI_API_KEY` in your shell or a local `.env`. `python‑dotenv` is available if you choose to load it.
- Use compose defaults unless the docs specify overrides. Document any new ports/credentials.

