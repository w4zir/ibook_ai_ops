.PHONY: setup start stop restart logs seed-data feast-apply test clean format deploy-prod

PYTHON ?= python

setup:
	@echo "Setting up Python environment (venv creation is optional)..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt
	@$(PYTHON) -m pip install -e .

start:
	@docker compose up -d

stop:
	@docker compose down

restart: stop start

logs:
	@docker compose logs -f

seed-data:
	@echo "Generating synthetic Feast data into data/processed/feast/..."
	@$(PYTHON) scripts/seed-data.py

feast-apply:
	@echo "Applying Feast feature repository (local DuckDB + Redis)..."
	@feast -c services/feast/feature_repo apply

test:
	@$(PYTHON) -m pytest tests/ -v --tb=short

clean:
	@docker compose down -v

format:
	@$(PYTHON) -m black common tests scripts || true
	@$(PYTHON) -m flake8 common tests scripts || true

deploy-prod:
	@echo "Phase 1: production deployment comes later (see scripts/deploy-production.sh in Phase 8)."
	@exit 0

