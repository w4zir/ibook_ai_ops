# Webook MLOps Platform - Implementation Plan
## Optimized for Cursor AI / Claude Code

---

## 🎯 Project Overview

**Goal:** Build a production-grade MLOps platform for Webook (Saudi Arabia's #1 ticketing platform) that can be developed locally and deployed to GKE.

**Tech Stack:**
- **Orchestration:** Docker Compose (local) → Kubernetes/GKE (production)
- **Feature Store:** Feast (Redis + DuckDB/BigQuery)
- **Experiment Tracking:** MLflow
- **Workflow:** Apache Airflow
- **Model Serving:** BentoML
- **Monitoring:** Prometheus + Grafana + Evidently AI
- **Data Quality:** Great Expectations
- **Version Control:** DVC

**Key Principle:** Same codebase for local development and production (config-driven differences only)

---

## 📁 Project Structure to Create

```
webook-mlops/
├── .github/
│   └── workflows/
│       └── mlops-cicd.yml
├── docker-compose.yml
├── .env.local
├── .env.production
├── .gitignore
├── Makefile
├── README.md
├── requirements.txt
├── requirements-dev.txt
│
├── services/
│   ├── mlflow/
│   │   ├── Dockerfile
│   │   └── .dockerignore
│   ├── feast/
│   │   ├── Dockerfile
│   │   ├── .dockerignore
│   │   └── feature_repo/
│   │       ├── feature_store.yaml
│   │       ├── __init__.py
│   │       ├── features.py
│   │       └── data_sources.py
│   ├── airflow/
│   │   ├── Dockerfile
│   │   ├── .dockerignore
│   │   ├── dags/
│   │   │   ├── __init__.py
│   │   │   ├── feature_engineering_pipeline.py
│   │   │   ├── model_training_pipeline.py
│   │   │   └── ml_monitoring_pipeline.py
│   │   └── plugins/
│   │       └── __init__.py
│   ├── bentoml/
│   │   ├── services/
│   │   │   ├── fraud_detection/
│   │   │   │   ├── Dockerfile
│   │   │   │   ├── bentofile.yaml
│   │   │   │   ├── service.py
│   │   │   │   └── model.py
│   │   │   ├── dynamic_pricing/
│   │   │   │   ├── Dockerfile
│   │   │   │   ├── bentofile.yaml
│   │   │   │   ├── service.py
│   │   │   │   └── model.py
│   │   │   └── recommendation/
│   │   │       ├── Dockerfile
│   │   │       ├── bentofile.yaml
│   │   │       ├── service.py
│   │   │       └── model.py
│   │   └── common/
│   │       ├── __init__.py
│   │       └── config.py
│   ├── monitoring/
│   │   ├── prometheus/
│   │   │   └── prometheus.yml
│   │   └── grafana/
│   │       ├── dashboards/
│   │       │   └── mlops-overview.json
│   │       └── datasources/
│   │           └── prometheus.yml
│   └── jupyter/
│       └── Dockerfile
│
├── scripts/
│   ├── setup-local.sh
│   ├── seed-data.py
│   ├── deploy-production.sh
│   ├── check-canary-metrics.py
│   └── init-db.sql
│
├── common/
│   ├── __init__.py
│   ├── config.py
│   ├── feature_utils.py
│   └── model_utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training_fraud.ipynb
│   └── 04_model_evaluation.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_features.py
│   │   ├── test_models.py
│   │   └── test_config.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_end_to_end_pipeline.py
│   │   └── test_api_endpoints.py
│   └── e2e/
│       ├── __init__.py
│       └── test_load_performance.py
│
├── terraform/
│   └── gcp/
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       ├── gke.tf
│       ├── storage.tf
│       ├── networking.tf
│       └── iam.tf
│
└── kubernetes/
    ├── base/
    │   ├── kustomization.yaml
    │   ├── namespace.yaml
    │   ├── mlflow-deployment.yaml
    │   ├── feast-deployment.yaml
    │   ├── airflow-deployment.yaml
    │   └── monitoring-stack.yaml
    └── overlays/
        ├── staging/
        │   └── kustomization.yaml
        └── production/
            ├── kustomization.yaml
            └── canary-5pct.yaml
```

---

## 🚀 Implementation Phases

### Phase 1: Local Environment Setup (Days 1-3)

#### 1.1 Initialize Project Structure

**AI Prompt for Cursor/Claude Code:**
```
Create the base project structure for webook-mlops with all directories and 
__init__.py files. Generate .gitignore for Python, Docker, and Terraform.
```

**Files to Create:**
- [ ] `.gitignore` (Python, Docker, Terraform, IDE files)
- [ ] `README.md` (project overview, quick start)
- [ ] `requirements.txt` (production dependencies)
- [ ] `requirements-dev.txt` (development dependencies)

**Dependencies to Include:**

`requirements.txt`:
```python
# MLOps Core
mlflow==2.10.2
feast[redis]==0.35.0
bentoml==1.2.3
apache-airflow==2.8.1

# Data Processing
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.4.0
xgboost==2.0.3
duckdb==0.9.2

# Monitoring
evidently==0.4.14
prometheus-client==0.19.0

# GCP
google-cloud-bigquery==3.15.0
google-cloud-storage==2.14.0
google-cloud-secret-manager==2.17.0

# Utilities
pydantic==2.5.3
python-dotenv==1.0.0
redis==5.0.1
psycopg2-binary==2.9.9
boto3==1.34.34
```

`requirements-dev.txt`:
```python
pytest==7.4.4
pytest-cov==4.1.0
pytest-asyncio==0.23.3
black==23.12.1
flake8==7.0.0
mypy==1.8.0
locust==2.20.0
jupyter==1.0.0
great-expectations==0.18.8
dvc==3.38.1
```

#### 1.2 Docker Compose Configuration

**AI Prompt:**
```
Create a docker-compose.yml file for local MLOps development with these services:
- PostgreSQL (for Airflow and MLflow metadata)
- Redis (for Feast online store)
- MinIO (S3-compatible storage)
- MLflow server
- Airflow (webserver + scheduler)
- Kafka + Zookeeper
- Prometheus + Grafana
- Jupyter notebook

Include health checks, volume mounts, and proper networking.
```

**Files to Create:**
- [ ] `docker-compose.yml`
- [ ] `.env.local` (local environment variables)
- [ ] `scripts/init-db.sql` (database initialization)

#### 1.3 Environment Configuration

**AI Prompt:**
```
Create a Python configuration module (common/config.py) that:
1. Loads environment variables from .env files
2. Provides environment-agnostic configuration (local vs production)
3. Handles Feast, MLflow, Airflow, and storage configurations
4. Uses Pydantic for validation
5. Supports both DuckDB (local) and BigQuery (production) for Feast offline store
```

**Files to Create:**
- [ ] `common/__init__.py`
- [ ] `common/config.py`
- [ ] `.env.local` (with all local configuration)
- [ ] `.env.production.example` (template for production)

#### 1.4 Makefile for Common Commands

**AI Prompt:**
```
Create a Makefile with these targets:
- setup: Initialize local environment
- start: Start all Docker services
- stop: Stop all services
- restart: Restart services
- logs: View logs
- seed-data: Generate sample data
- feast-apply: Apply Feast feature definitions
- test: Run test suite
- clean: Remove containers and volumes
- format: Run black and flake8
- deploy-prod: Deploy to production
```

**Files to Create:**
- [ ] `Makefile`

---

### Phase 2: Feature Store Setup (Days 4-6)

#### 2.1 Feast Feature Definitions

**AI Prompt:**
```
Create Feast feature definitions for a ticketing platform with:

1. Entities: event, user, promoter
2. Feature views:
   - event_realtime_metrics (current_inventory, sell_through_rate_5min, concurrent_viewers)
   - event_historical_metrics (total_tickets_sold, avg_ticket_price, promoter_success_rate)
   - user_purchase_behavior (lifetime_purchases, fraud_risk_score, preferred_category)
   
3. Data sources:
   - Local: FileSource (Parquet files)
   - Production: BigQuerySource (conditional based on ENVIRONMENT variable)

4. Configuration:
   - feature_store.yaml with local (DuckDB) and production (BigQuery) settings
   - Online store: Redis (both environments)
```

**Files to Create:**
- [ ] `services/feast/feature_repo/feature_store.yaml`
- [ ] `services/feast/feature_repo/features.py`
- [ ] `services/feast/feature_repo/data_sources.py`
- [ ] `services/feast/Dockerfile`

#### 2.2 Sample Data Generation

**AI Prompt:**
```
Create a Python script (scripts/seed-data.py) that generates synthetic ticketing data:
- 10,000 transactions
- 100 events across categories (sports, concerts, family, cultural)
- 1,000 users with various purchase patterns
- 3% fraud rate
- Seasonal patterns and trends

Output format: Parquet files for Feast offline store
Columns needed for fraud detection, pricing, and recommendations
```

**Files to Create:**
- [ ] `scripts/seed-data.py`

#### 2.3 Feature Utilities

**AI Prompt:**
```
Create utility functions (common/feature_utils.py) for:
1. Fetching online features from Feast (with caching)
2. Creating training datasets with point-in-time correctness
3. Feature validation and monitoring
4. Feature store health checks
```

**Files to Create:**
- [ ] `common/feature_utils.py`

---

### Phase 3: MLflow Setup (Days 7-9)

#### 3.1 MLflow Server Configuration

**AI Prompt:**
```
Create MLflow Dockerfile that:
1. Uses PostgreSQL backend (local) or Cloud SQL (production)
2. Uses MinIO (local) or GCS (production) for artifacts
3. Includes health check endpoint
4. Configures authentication (optional)
```

**Files to Create:**
- [ ] `services/mlflow/Dockerfile`

#### 3.2 Model Training Scripts

**AI Prompt:**
```
Create a fraud detection model training script that:
1. Loads features from Feast
2. Trains an XGBoost classifier
3. Logs experiments to MLflow (params, metrics, artifacts)
4. Registers the model in MLflow registry
5. Includes hyperparameter tuning with Optuna
6. Saves SHAP explainer for interpretability

Include proper error handling and logging.
```

**Files to Create:**
- [ ] `notebooks/03_model_training_fraud.ipynb`
- [ ] `common/model_utils.py` (shared training utilities)

---

### Phase 4: Model Serving with BentoML (Days 10-14)

#### 4.1 Fraud Detection Service

**AI Prompt:**
```
Create a BentoML service for fraud detection that:
1. Loads the latest model from MLflow registry
2. Fetches real-time features from Feast
3. Supports batch prediction (adaptive batching)
4. Returns fraud score + SHAP explanation
5. Includes Prometheus metrics export
6. Has health check and readiness endpoints

Service should work identically in local Docker and production Kubernetes.
```

**Files to Create:**
- [ ] `services/bentoml/services/fraud_detection/service.py`
- [ ] `services/bentoml/services/fraud_detection/model.py`
- [ ] `services/bentoml/services/fraud_detection/bentofile.yaml`
- [ ] `services/bentoml/services/fraud_detection/Dockerfile`

#### 4.2 Dynamic Pricing Service

**AI Prompt:**
```
Create a BentoML service for dynamic pricing that:
1. Implements reinforcement learning-based pricing (Thompson Sampling)
2. Fetches features: current inventory, demand velocity, competitor prices
3. Returns recommended price + confidence interval
4. Supports A/B testing with multi-armed bandit
5. Includes circuit breaker for fallback to rule-based pricing

Optimize for <100ms p99 latency.
```

**Files to Create:**
- [ ] `services/bentoml/services/dynamic_pricing/service.py`
- [ ] `services/bentoml/services/dynamic_pricing/model.py`
- [ ] `services/bentoml/services/dynamic_pricing/bentofile.yaml`
- [ ] `services/bentoml/services/dynamic_pricing/Dockerfile`

#### 4.3 Common BentoML Utilities

**AI Prompt:**
```
Create shared utilities for BentoML services:
1. Feature fetching wrapper (with caching)
2. Model loading from MLflow
3. Prometheus metrics helpers
4. Request/response validation with Pydantic
5. Error handling and logging
```

**Files to Create:**
- [ ] `services/bentoml/common/config.py`
- [ ] `services/bentoml/common/feast_client.py`
- [ ] `services/bentoml/common/mlflow_client.py`
- [ ] `services/bentoml/common/metrics.py`

---

### Phase 5: Airflow Workflows (Days 15-18)

#### 5.1 Feature Engineering Pipeline

**AI Prompt:**
```
Create an Airflow DAG (feature_engineering_pipeline.py) that:
1. Runs hourly
2. Aggregates real-time data from Kafka
3. Computes batch features in DuckDB (local) or BigQuery (production)
4. Validates feature quality with Great Expectations
5. Materializes features to Feast online/offline stores
6. Triggers model retraining if drift detected

Include proper error handling, retries, and alerting.
```

**Files to Create:**
- [ ] `services/airflow/dags/feature_engineering_pipeline.py`
- [ ] `services/airflow/Dockerfile`

#### 5.2 Model Training Pipeline

**AI Prompt:**
```
Create an Airflow DAG (model_training_pipeline.py) that:
1. Runs weekly (or triggered by drift detection)
2. Fetches training data from Feast
3. Trains model on Kubernetes pod (using KubernetesPodOperator for production)
4. Evaluates model against baseline
5. Registers model in MLflow if improvement > threshold
6. Deploys to staging (canary)
7. Monitors canary for 24 hours
8. Promotes to production if successful

Include branching logic for quality gates.
```

**Files to Create:**
- [ ] `services/airflow/dags/model_training_pipeline.py`

#### 5.3 ML Monitoring Pipeline

**AI Prompt:**
```
Create an Airflow DAG (ml_monitoring_pipeline.py) that:
1. Runs daily
2. Generates Evidently AI drift reports for all production models
3. Compares predictions to actuals
4. Triggers retraining if:
   - Data drift > 30%
   - Model performance degrades > 10%
5. Sends alerts to Slack/PagerDuty

Store drift reports in GCS for historical analysis.
```

**Files to Create:**
- [ ] `services/airflow/dags/ml_monitoring_pipeline.py`

---

### Phase 6: Monitoring & Observability (Days 19-21)

#### 6.1 Prometheus Configuration

**AI Prompt:**
```
Create Prometheus configuration that scrapes:
1. BentoML model serving metrics (latency, throughput, errors)
2. MLflow server metrics
3. Feast feature store metrics
4. Airflow DAG metrics
5. Custom business metrics (revenue impact, fraud savings)

Include alert rules for:
- Model latency p99 > 200ms
- Error rate > 1%
- Feature freshness > 5 minutes
- Data drift detected
```

**Files to Create:**
- [ ] `services/monitoring/prometheus/prometheus.yml`
- [ ] `services/monitoring/prometheus/alert_rules.yml`

#### 6.2 Grafana Dashboards

**AI Prompt:**
```
Create Grafana dashboard JSON for MLOps overview showing:
1. Model serving latency (p50, p95, p99) by service
2. Request rate and error rate
3. Feature store query performance
4. Airflow DAG success rate
5. Business metrics (pricing revenue impact, fraud savings)
6. Data drift scores over time

Use Prometheus as data source.
```

**Files to Create:**
- [ ] `services/monitoring/grafana/dashboards/mlops-overview.json`
- [ ] `services/monitoring/grafana/datasources/prometheus.yml`

#### 6.3 Evidently AI Integration

**AI Prompt:**
```
Create Python utilities for ML monitoring with Evidently AI:
1. Generate drift reports (data drift, concept drift, prediction drift)
2. Compare model performance over time
3. Create HTML reports and save to GCS
4. Extract metrics for Prometheus
5. Trigger alerts when thresholds exceeded

Support for both batch and real-time monitoring.
```

**Files to Create:**
- [ ] `common/monitoring_utils.py`

---

### Phase 7: Testing Framework (Days 22-25)

#### 7.1 Unit Tests

**AI Prompt:**
```
Create pytest unit tests for:
1. Feature engineering functions (test_features.py)
2. Model training utilities (test_models.py)
3. Configuration loading (test_config.py)
4. Feature store operations (test_feast_utils.py)

Aim for >90% code coverage.
Use pytest fixtures for Feast and MLflow setup.
Mock external dependencies.
```

**Files to Create:**
- [ ] `tests/conftest.py` (shared fixtures)
- [ ] `tests/unit/test_features.py`
- [ ] `tests/unit/test_models.py`
- [ ] `tests/unit/test_config.py`
- [ ] `tests/unit/test_feast_utils.py`

#### 7.2 Integration Tests

**AI Prompt:**
```
Create integration tests that:
1. Start Docker Compose services
2. Wait for health checks
3. Test end-to-end pipeline:
   - Generate data → Feast → Model training → MLflow → Serving
4. Test API endpoints (BentoML services)
5. Validate Airflow DAG execution
6. Check monitoring metrics

Use pytest-docker-compose plugin.
```

**Files to Create:**
- [ ] `tests/integration/test_end_to_end_pipeline.py`
- [ ] `tests/integration/test_api_endpoints.py`

#### 7.3 Load Testing

**AI Prompt:**
```
Create Locust load tests for BentoML services:
1. Fraud detection: 1000 RPS sustained
2. Dynamic pricing: 500 RPS sustained
3. Validate p99 latency < 200ms
4. Check error rate < 0.1%

Include various payload sizes and traffic patterns.
```

**Files to Create:**
- [ ] `tests/e2e/test_load_performance.py`

---

### Phase 8: CI/CD Pipeline (Days 26-28)

#### 8.1 GitHub Actions Workflow

**AI Prompt:**
```
Create GitHub Actions workflow (.github/workflows/mlops-cicd.yml) that:

On Pull Request:
1. Run linting (black, flake8, mypy)
2. Run unit tests with coverage
3. Start Docker Compose for integration tests
4. Build Docker images

On Main Branch Push:
1. All PR checks
2. Build and push images to GCR
3. Deploy to staging namespace
4. Run smoke tests
5. Manual approval for production
6. Deploy to production with canary rollout
7. Monitor canary for 1 hour
8. Promote to 100% or rollback

Include job dependencies and proper secrets handling.
```

**Files to Create:**
- [ ] `.github/workflows/mlops-cicd.yml`

#### 8.2 Deployment Scripts

**AI Prompt:**
```
Create bash script (scripts/deploy-production.sh) that:
1. Authenticates with GCP
2. Gets GKE credentials
3. Builds Docker images with git SHA tag
4. Pushes to Google Container Registry
5. Applies Kubernetes manifests (using Kustomize)
6. Updates image tags in deployments
7. Waits for rollout completion
8. Runs smoke tests
9. Reports success/failure

Include error handling and rollback on failure.
```

**Files to Create:**
- [ ] `scripts/deploy-production.sh`
- [ ] `scripts/check-canary-metrics.py`
- [ ] `scripts/rollback-deployment.sh`

---

### Phase 9: Infrastructure as Code (Days 29-32)

#### 9.1 Terraform for GCP

**AI Prompt:**
```
Create Terraform configuration for GCP production infrastructure:

1. GKE cluster (gke.tf):
   - Region: me-central2 (Saudi Arabia)
   - Node pools: general-purpose, model-serving, feature-store
   - Autoscaling: 3-100 nodes
   - Workload Identity enabled
   - Private cluster configuration

2. Storage (storage.tf):
   - GCS buckets: artifacts, data, models
   - Cloud SQL: PostgreSQL for MLflow/Airflow
   - Memorystore: Redis cluster for Feast

3. Networking (networking.tf):
   - VPC with private subnets
   - Cloud NAT for outbound traffic
   - Firewall rules

4. IAM (iam.tf):
   - Service accounts with least privilege
   - Workload Identity bindings

Include variables.tf and outputs.tf.
```

**Files to Create:**
- [ ] `terraform/gcp/main.tf`
- [ ] `terraform/gcp/variables.tf`
- [ ] `terraform/gcp/outputs.tf`
- [ ] `terraform/gcp/gke.tf`
- [ ] `terraform/gcp/storage.tf`
- [ ] `terraform/gcp/networking.tf`
- [ ] `terraform/gcp/iam.tf`

#### 9.2 Kubernetes Manifests

**AI Prompt:**
```
Create Kubernetes manifests with Kustomize:

Base resources (kubernetes/base/):
1. Namespace definitions
2. MLflow deployment + service
3. Feast deployment + service
4. Airflow deployment (webserver + scheduler)
5. BentoML deployments for each model
6. Redis deployment (if not using Memorystore)
7. Prometheus + Grafana stack

Overlays:
- Staging: 2 replicas, smaller resources
- Production: 3+ replicas, HPA, resource limits

Include ConfigMaps, Secrets (from Secret Manager), PVCs, and NetworkPolicies.
```

**Files to Create:**
- [ ] `kubernetes/base/kustomization.yaml`
- [ ] `kubernetes/base/namespace.yaml`
- [ ] `kubernetes/base/mlflow-deployment.yaml`
- [ ] `kubernetes/base/feast-deployment.yaml`
- [ ] `kubernetes/base/airflow-deployment.yaml`
- [ ] `kubernetes/base/bentoml-deployments.yaml`
- [ ] `kubernetes/overlays/staging/kustomization.yaml`
- [ ] `kubernetes/overlays/production/kustomization.yaml`
- [ ] `kubernetes/overlays/production/canary-5pct.yaml`

---

### Phase 10: Documentation & Polish (Days 33-35)

#### 10.1 Documentation

**AI Prompt:**
```
Create comprehensive README.md with:
1. Project overview and architecture diagram
2. Quick start guide (local development)
3. Prerequisites and installation
4. Usage examples for each service
5. Testing instructions
6. Deployment guide
7. Troubleshooting common issues
8. Contributing guidelines
```

**Files to Create:**
- [ ] `README.md`
- [ ] `docs/architecture.md`
- [ ] `docs/local-development.md`
- [ ] `docs/production-deployment.md`
- [ ] `docs/troubleshooting.md`

#### 10.2 Runbooks

**AI Prompt:**
```
Create operational runbooks for:
1. Incident response (model latency spike, accuracy drop)
2. Scaling procedures (traffic surge during mega-events)
3. Disaster recovery (data loss, service outage)
4. Model rollback procedures
5. Feature store corruption recovery
```

**Files to Create:**
- [ ] `docs/runbooks/incident-response.md`
- [ ] `docs/runbooks/scaling.md`
- [ ] `docs/runbooks/disaster-recovery.md`

---

## 📋 Development Checklist

### Week 1: Foundation
- [ ] Initialize Git repository
- [ ] Set up project structure
- [ ] Create Docker Compose environment
- [ ] Implement configuration system
- [ ] Generate sample data
- [ ] Verify all services start successfully

### Week 2: Feature Store
- [ ] Define Feast features
- [ ] Implement feature engineering functions
- [ ] Test feature retrieval (online/offline)
- [ ] Verify DuckDB → BigQuery compatibility

### Week 3: Model Development
- [ ] Set up MLflow tracking
- [ ] Train fraud detection model
- [ ] Create model registry workflow
- [ ] Implement model versioning

### Week 4: Model Serving
- [ ] Build BentoML fraud service
- [ ] Test local serving
- [ ] Add monitoring metrics
- [ ] Validate <200ms latency

### Week 5: Orchestration
- [ ] Create Airflow DAGs
- [ ] Test feature engineering pipeline
- [ ] Implement training pipeline
- [ ] Add monitoring pipeline

### Week 6: Testing & CI/CD
- [ ] Write unit tests (>90% coverage)
- [ ] Create integration tests
- [ ] Set up GitHub Actions
- [ ] Test full CI/CD flow

### Week 7: Infrastructure
- [ ] Write Terraform configs
- [ ] Create Kubernetes manifests
- [ ] Test staging deployment
- [ ] Validate production readiness

### Week 8: Production Launch
- [ ] Deploy to production
- [ ] Monitor canary rollout
- [ ] Validate all services
- [ ] Create runbooks

---

## 🤖 Cursor/Claude Code AI Prompts

### For File Generation

When creating a new component, use this template:

```
Create a [component type] for webook MLOps platform that:

Requirements:
1. [Functional requirement 1]
2. [Functional requirement 2]
3. [Non-functional requirement - performance/security]

Technical specs:
- Language: Python 3.11
- Framework: [FastAPI/BentoML/Airflow/etc]
- Dependencies: [list key dependencies]
- Environment: Works in both local (Docker) and production (GKE)

Integration points:
- Connects to: [Feast/MLflow/etc]
- Exposes: [API/metrics/logs]

Code style:
- Use type hints
- Include docstrings (Google style)
- Add error handling
- Include logging
- Write defensive code

Example usage:
[Provide example of how this will be called]
```

### For Debugging

```
I'm getting this error in [component]:
[paste error]

Context:
- Environment: [local/staging/production]
- Service: [service name]
- Recent changes: [what you changed]

Help me:
1. Understand the root cause
2. Fix the issue
3. Prevent it in the future
```

### For Code Review

```
Review this code for:
1. Best practices adherence
2. Performance optimization
3. Security vulnerabilities
4. Error handling
5. Testability
6. Production readiness

[paste code]
```

### For Refactoring

```
Refactor this code to:
1. Improve readability
2. Reduce duplication
3. Enhance performance
4. Add type safety
5. Make it more testable

Current code:
[paste code]

Keep the same functionality but make it production-grade.
```

---

## 🎯 Success Criteria

### Local Development
- [ ] All services start with `make start`
- [ ] Sample data generates successfully
- [ ] Features can be fetched from Feast
- [ ] Model training completes in <10 minutes
- [ ] Predictions work via BentoML API
- [ ] All tests pass locally

### Production Deployment
- [ ] Infrastructure deploys via Terraform
- [ ] Services deploy to GKE successfully
- [ ] Health checks pass
- [ ] Monitoring dashboards show data
- [ ] Can handle 1000+ RPS
- [ ] p99 latency < 200ms
- [ ] Zero-downtime deployments work

### Business Outcomes
- [ ] Fraud detection accuracy > 95%
- [ ] Dynamic pricing revenue uplift > 5%
- [ ] Feature freshness < 1 minute
- [ ] Model retraining automated
- [ ] Drift detection working
- [ ] Canary rollouts validated

---

## 🔧 Troubleshooting Guide

### Common Issues & Solutions

**Issue:** Docker services won't start
```bash
# Solution
docker-compose down -v
docker system prune -f
make clean && make start
```

**Issue:** Feast can't connect to Redis
```bash
# Solution
# Check Redis password in .env.local
docker logs webook-redis
docker exec -it webook-redis redis-cli ping
```

**Issue:** MLflow artifacts not saving
```bash
# Solution
# Verify MinIO is running and bucket exists
docker exec -it webook-minio mc ls local/
```

**Issue:** BentoML can't load model
```bash
# Solution
# Check MLflow model registry
curl http://localhost:5000/api/2.0/mlflow/registered-models/list
# Verify model version exists
```

**Issue:** Airflow DAG not running
```bash
# Solution
# Check Airflow logs
docker logs webook-airflow-scheduler
# Manually trigger DAG
curl -X POST http://localhost:8080/api/v1/dags/{dag_id}/dagRuns \
  -u admin:admin -H "Content-Type: application/json" -d '{}'
```

---

## 📚 Learning Resources

### Documentation
- **Feast:** https://docs.feast.dev
- **MLflow:** https://mlflow.org/docs/latest/index.html
- **BentoML:** https://docs.bentoml.org
- **Airflow:** https://airflow.apache.org/docs
- **Evidently AI:** https://docs.evidentlyai.com

### Tutorials
- **Feast Quickstart:** https://docs.feast.dev/getting-started/quickstart
- **MLflow Tracking:** https://mlflow.org/docs/latest/tracking.html
- **BentoML Tutorial:** https://docs.bentoml.org/en/latest/tutorial.html
- **Airflow Tutorial:** https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html

### Best Practices
- **12-Factor App:** https://12factor.net
- **ML Ops Best Practices:** https://ml-ops.org
- **Google SRE Book:** https://sre.google/sre-book/table-of-contents/

---

## 🎓 Next Steps After Implementation

1. **Expand ML Use Cases:**
   - Recommendation system
   - Demand forecasting
   - Customer segmentation
   - Queue optimization

2. **Advanced Features:**
   - A/B testing framework
   - Automated hyperparameter tuning
   - Online learning capabilities
   - Federated learning for privacy

3. **Scale Optimization:**
   - GPU support for deep learning
   - Distributed training with Ray
   - Feature store performance tuning
   - Cost optimization

4. **Governance:**
   - Model explainability dashboard
   - Bias detection and mitigation
   - PDPL/GDPR automation
   - Audit logging

---

## 📞 Support

For issues or questions:
1. Check troubleshooting guide above
2. Review service logs: `make logs`
3. Run tests: `make test`
4. Consult documentation in `docs/`

---

**Last Updated:** February 2026  
**Maintainer:** ML Platform Team  
**Status:** Ready for Implementation
