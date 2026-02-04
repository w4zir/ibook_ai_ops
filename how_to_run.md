## How to run (Phases 1–2)

This repo currently covers **Phase 1–2**:

- Phase 1: local service stack + config module + basic unit tests.
- Phase 2: Feast feature repository + synthetic sample data + feature utilities.

### Prerequisites

- **Docker Desktop** (with Compose)
- **Python** 3.10+ (this repo was tested with Python 3.12)
- **Git**

Optional:
- **WSL / Git Bash** if you want to use the provided `Makefile`

---

## One-time setup (PowerShell)

From repo root (`d:\\ai_ws\\projects\\ibook_ai_ops`):

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\pip install -e .
```

Notes:
- `.env.local` is committed with safe local defaults. Update values if ports or credentials conflict on your machine.

---

## Start the local stack

### PowerShell (recommended on Windows)

```bash
docker compose up -d --build
docker compose ps
```

Stop:

```bash
docker compose down
```

Clean (also removes volumes):

```bash
docker compose down -v
```

### Make (WSL / Git Bash / Linux / macOS)

```bash
make setup
make start
make logs
make stop
```

---

## Service URLs (defaults)

- **MLflow**: `http://localhost:5000`
- **Airflow**: `http://localhost:8080` (default user/pass: `admin` / `admin`)
- **MinIO API**: `http://localhost:9000`
- **MinIO console**: `http://localhost:9001` (default user/pass: `minioadmin` / `minioadmin`)
- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (default user/pass: `admin` / `admin`)
- **Jupyter**: `http://localhost:8888` (token disabled in Phase 1 container)

---

## Run tests (no Docker required)

### PowerShell

```bash
.venv\Scripts\python -m pytest tests\ -v --tb=short
```

### Make

```bash
make test
```

---

## Phase 2: Feature store & sample data

Phase 2 adds:
- A Feast feature repo in `services/feast/feature_repo/`.
- Synthetic Parquet datasets under `data/processed/feast/` via `scripts/seed-data.py`.
- Convenience helpers in `common/feature_utils.py`.

### Generate synthetic data

From the repo root:

#### PowerShell

```bash
.venv\Scripts\python scripts\seed-data.py
```

#### Make (WSL / Git Bash / Linux / macOS)

```bash
make seed-data
```

This will create:
- `data/processed/feast/event_metrics.parquet`
- `data/processed/feast/user_metrics.parquet`

### Apply the Feast feature repo (local)

After generating data, apply the Feast definitions so the registry is created:

```bash
feast -c services/feast/feature_repo apply
```

If you prefer using `make`:

```bash
make feast-apply
```

### Example: fetch online features (Python)

Once Redis and the local stack are running (`docker compose up -d`), you can
experiment with online features from a Python REPL or notebook:

```python
from common.feature_utils import fetch_online_features

rows = [{"event_id": 1}]
features = ["event_realtime_metrics:current_inventory"]

df = fetch_online_features(features=features, entity_rows=rows)
print(df)
```

---

## Troubleshooting

- **Ports already in use**: stop conflicting local services or change the exposed ports in `docker-compose.yml`.
- **Reset everything**:

```bash
docker compose down -v
docker compose up -d --build
```

