# Georgian Tourism Agent

Multi-agent FastAPI service for planning travel itineraries in Georgia.

The service uses deterministic LangGraph routing, direct Qdrant search,
GeoAgent enrichment before planning, one-call itinerary planning,
programmatic validation, response formatting, optional evaluation, and memory
save/load hooks.

## Pipeline

For planning requests:

```text
memory_load -> orchestrator_plan -> search_agent -> geo_agent -> planning_agent -> validation_agent -> response_agent -> eval_node -> memory_save
```

For search/info requests:

```text
memory_load -> orchestrator_plan -> search_agent -> response_agent -> eval_node -> memory_save
```

## Setup

Install runtime and test dependencies into a virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Environment variables are documented in `ENVIRONMENT.md`. The application does
not load `.env` automatically; variables must be present in the process
environment.

## Run

```bash
.venv/bin/python main.py
```

The API starts on `http://0.0.0.0:8000`.

Useful endpoints:

- `GET /api/v1/health`
- `POST /api/v1/plan`
- `GET /docs`
- `GET /metrics`

## Tests

Use this command locally to avoid loading unrelated system pytest plugins:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m 'not online'
```

Disabling plugin autoload is required to skip a system pytest plugin that breaks
collection. `pyproject.toml` force-loads `pytest-asyncio` via `addopts`, so async
tests still run (and are not silently skipped) under this command.

Online tests are skipped by default. They call external services and may spend
provider credits:

```bash
RUN_ONLINE_TESTS=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q -m online
```

## Repository Notes

Local secrets, virtual environments, IDE metadata, checkpoints, caches, and
generated artifacts are excluded through `.gitignore`.
