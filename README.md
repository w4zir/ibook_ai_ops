## Webook MLOps Platform (Phase 1)

Production-grade MLOps platform for Webook (ticketing) designed to run locally via Docker Compose and later deploy to Kubernetes/GKE.

### What’s included in Phase 1

- Local project skeleton + dependency manifests
- Local orchestration via `docker-compose.yml` (Postgres, Redis, MinIO, MLflow, Airflow, Kafka, Prometheus, Grafana, Jupyter)
- Environment configuration module in `common/config.py`
- A small unit test suite for config loading/validation

### Quick start

- See `how_to_run.md` for:
  - Windows PowerShell commands (no `make` required)
  - Makefile usage (WSL/Git Bash/Linux/macOS)
  - Service URLs, testing, and troubleshooting

