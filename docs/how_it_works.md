# How It Works

This document describes each component of the Ibook MLOps platform: architecture, data flows, algorithms (in pseudo-code style), and configuration.

---

## 1. System Architecture Overview

The platform runs as a set of services orchestrated by Docker Compose (local) or Kubernetes (production). Data flows from events and users through the feature store into model training and serving, with monitoring and drift detection closing the loop.

```mermaid
flowchart TB
    subgraph infra [Infrastructure]
        Postgres[(PostgreSQL)]
        Redis[(Redis)]
        MinIO[(MinIO)]
        Kafka[Kafka]
        Zookeeper[Zookeeper]
    end

    subgraph platform [MLOps Platform]
        Feast[Feast Feature Store]
        MLflow[MLflow]
        Airflow[Airflow]
        BentoFraud[BentoML Fraud]
        BentoPricing[BentoML Pricing]
    end

    subgraph observability [Observability]
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    Airflow --> Postgres
    Airflow --> Feast
    Airflow --> MLflow
    MLflow --> Postgres
    MLflow --> MinIO
    Feast --> Redis
    BentoFraud --> MLflow
    BentoFraud --> Feast
    BentoPricing --> Feast
    BentoFraud --> Prometheus
    BentoPricing --> Prometheus
    Prometheus --> Grafana
```

**Data flow (simplified):**

1. **Training:** Airflow (or notebook) builds a training dataset from Feast (or synthetic data) → trains XGBoost with Optuna → logs model and metrics to MLflow.
2. **Serving:** BentoML fraud/pricing services load the model from MLflow and features from Feast → respond to prediction/recommend requests.
3. **Monitoring:** Prometheus scrapes BentoML (and optionally MLflow/Airflow) → Grafana visualizes; Airflow runs a daily monitoring DAG that computes drift and can trigger retraining.

---

## 2. Infrastructure Layer

| Component | Role |
|-----------|------|
| **PostgreSQL** | Shared by Airflow (DAG metadata, task state) and MLflow (experiments, runs, model registry). Databases: `airflow`, `mlflow`. |
| **Redis** | Feast online store: low-latency feature lookup for real-time prediction. |
| **MinIO** | S3-compatible object storage; MLflow stores model artifacts and SHAP outputs here. |
| **Kafka + Zookeeper** | Event streaming. Used by the feature engineering DAG in production; locally the pipeline uses stubs. |

---

## 3. Feature Store (Feast)

**Location:** `services/feast/feature_repo/`

**Structure:**

- `feature_store.yaml` — store config (offline: DuckDB local / BigQuery prod; online: Redis).
- `data_sources.py` — data source definitions (e.g. Parquet file sources for local).
- `features.py` — entities and feature views.

**Entities:** `event` (event_id), `user` (user_id), `promoter` (promoter_id).

**Feature views:**

- **event_realtime_metrics** — current_inventory, sell_through_rate_5min, concurrent_viewers (TTL 60 min).
- **event_historical_metrics** — total_tickets_sold, avg_ticket_price, promoter_success_rate (TTL 365 days).
- **user_purchase_behavior** — lifetime_purchases, fraud_risk_score, preferred_category (TTL 365 days).

**Online vs offline:** Online = Redis (low-latency serving). Offline = DuckDB (local) or BigQuery (production) for historical training datasets.

**Feature retrieval (pseudo-code):**

```
FUNCTION fetch_online_features(features[], entity_rows[], store, use_cache):
    fs = store OR get_feature_store(repo_path)
    IF use_cache AND cache.has(fs, features, entity_rows):
        RETURN cache.get(fs, features, entity_rows)
    response = fs.get_online_features(features=features, entity_rows=entity_rows)
    df = response.to_df()
    IF use_cache:
        cache.put(fs, features, entity_rows, df)
    RETURN df
```

Training datasets use **historical retrieval** with an entity DataFrame that includes `event_timestamp` for point-in-time correctness: `fs.get_historical_features(entity_df=entity_df, features=feature_refs)`.

---

## 4. Model Training Pipeline

**Algorithm:** XGBoost binary classifier for fraud detection (objective: `binary:logistic`).

**Implementation:** `common/model_utils.py` — `build_fraud_training_dataframe()` and `train_fraud_model()`.

**Training data:** Features: `lifetime_purchases`, `fraud_risk_score`. Label: `is_fraud_label` = 1 if `fraud_risk_score >= fraud_threshold` (default 0.05), else 0.

**Pseudo-code: training loop**

```
FUNCTION train_fraud_model(df, target_column, n_trials=5, test_size=0.2):
    feature_names = resolve_feature_columns(df, target_column)
    X, y = df[feature_names], df[target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size, stratify=y)

    study = create_study(direction="maximize")
    FOR trial IN 1..n_trials:
        params = {
            max_depth: suggest_int(2, 4),
            learning_rate: suggest_float_log(0.05, 0.3),
            n_estimators: suggest_int(20, 60),
            subsample: suggest_float(0.6, 1.0),
            colsample_bytree: suggest_float(0.6, 1.0)
        }
        model = XGBClassifier(**params, objective="binary:logistic")
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred_proba)
        log_params_and_metrics_to_mlflow(params, auc, accuracy)
        RETURN auc  // Optuna maximizes AUC

    best_params = study.best_params
    best_model = XGBClassifier(**best_params).fit(X_train, y_train)
    SHAP: explainer = TreeExplainer(best_model); log shap values + sample to MLflow artifacts
    mlflow.xgboost.log_model(best_model, artifact_path="model")
    RETURN TrainingResult(model, feature_names, best_params, roc_auc, accuracy, run_id)
```

**MLflow:** Logs parameters, ROC-AUC and accuracy, model artifact, and SHAP artifacts under experiment `fraud_detection`.

**Airflow DAG:** `model_training_pipeline` runs on a weekly schedule (`@weekly`). Tasks: build training dataset → train_model → evaluate_against_baseline → register_and_mark_candidate → deploy_canary (stub) → monitor_canary (stub) → finalize_promotion (stub).

---

## 5. Model Serving (BentoML)

### 5.1 Fraud Detection Service (port 7001)

**Flow:** HTTP batch request → build entity rows (user_id, event_id) → fetch Feast online features (e.g. user_purchase_behavior: lifetime_purchases, fraud_risk_score) → run XGBoost inference → return fraud_score and is_fraud (threshold 0.5).

**Pseudo-code:**

```
ON POST /predict WITH FraudBatchRequest:
    FOR each request IN batch.requests:
        entity_rows.append({user_id, event_id})
    features_df = get_fraud_features_for_entities(entity_rows)  // Feast + optional overrides
    predictions = model.predict_batch(features_df)  // XGBoost predict_proba; threshold 0.5
    RETURN FraudBatchResponse(predictions)
```

**Model loading:** Resolves latest model from MLflow registry (by stage or version), then loads the XGBoost artifact.

### 5.2 Dynamic Pricing Service (port 7002)

**Algorithm:** Thompson Sampling multi-armed bandit with Beta(α, β) per arm. Arms in code: `baseline`, `aggressive` (and optionally fallback for circuit breaker).

**Pseudo-code: Thompson Sampling**

```
STATE: arms = { "baseline": (α, β), "aggressive": (α, β) }  // Beta priors

FUNCTION choose_arm():
    FOR each arm, (α, β) IN arms:
        sample[arm] = random_beta(α, β)
    RETURN argmax(sample)

FUNCTION update(arm, reward):   // reward in [0, 1]
    α, β = arms[arm]
    α += clip(reward, 0, 1)
    β += 1 - clip(reward, 0, 1)
    arms[arm] = (α, β)
```

**Pricing logic (pseudo-code):**

```
FUNCTION recommend_price(request, features_row):
    arm = bandit.choose_arm()
    base_price = request.current_price
    IF arm == "aggressive": uplift = 0.10 ELSE uplift = 0.0
    velocity = features_row["event_realtime_metrics__sell_through_rate_5min"] OR 1.0
    inventory = features_row["event_realtime_metrics__current_inventory"] OR 1.0
    demand_factor = clip(velocity / max(inventory, 1), 0.5, 2.0)
    recommended = base_price * (1 + uplift) * demand_factor
    RETURN (recommended, arm, lower_ci, upper_ci)
```

**Fallback:** `rule_based_fallback(current_price)` returns `current_price * 1.05` when the bandit or features are unavailable (circuit breaker).

---

## 6. Drift Detection and Retraining

**Implementation:** `common/monitoring_utils.py` — Evidently AI `DataDriftPreset` when available; otherwise a statistical fallback.

**Fallback drift score (pseudo-code):**

```
FUNCTION fallback_drift_score(reference_df, current_df):
    common_cols = numeric columns present in both DataFrames
    FOR col IN common_cols:
        ref_mean, cur_mean = reference_df[col].mean(), current_df[col].mean()
        std = reference_df[col].std() OR 1.0
        column_scores[col] = min(1.0, |cur_mean - ref_mean| / (std + ε))
    drift_score = mean(column_scores)
    drift_detected = (drift_score >= 0.3)
    RETURN DriftResult(drift_score, drift_detected, column_scores)
```

**Threshold:** 0.3 (30%). When `drift_score >= 0.3`, the system considers drift significant and can trigger alerts and retraining.

**Airflow DAG:** `ml_monitoring_pipeline` runs daily (`@daily`). Tasks: collect_production_metrics → compute_drift_reports → check_thresholds → send_alerts (stub) and trigger_retraining (stub). The trigger task logs that it would trigger `model_training_pipeline`; in production this would use `TriggerDagRunOperator`.

```mermaid
flowchart LR
    Collect[collect_production_metrics] --> Compute[compute_drift_reports]
    Compute --> Check[check_thresholds]
    Check --> Alert[send_alerts]
    Check --> Retrain[trigger_retraining]
```

### 6.1 Auto-Training on Failure Rate (Real-time Closed Loop)

In addition to the batch-oriented Airflow monitoring, the platform supports **real-time auto-retraining** triggered by prediction failure rate. This is the staff-engineering approach to continuous model improvement: a closed-loop system that detects model degradation in production and automatically retrains and hot-reloads a new model without human intervention.

**Architecture:**

```mermaid
flowchart TB
    subgraph serving [BentoML Fraud Service]
        Predict["/predict endpoint"]
        Feedback["/feedback endpoint"]
        Tracker[FailureTracker<br/>sliding window]
        Runtime["FraudModelRuntime<br/>(hot-swappable)"]
    end

    subgraph retrain [Auto-Training Pipeline]
        Orch[AutoTrainingOrchestrator]
        Build[Build dataset]
        Train[Train XGBoost + Optuna]
        Eval[Evaluate vs baseline]
        Register[Register in MLflow]
        Reload[Hot-reload model]
    end

    Predict --> Runtime
    Feedback --> Tracker
    Tracker -->|"failure_rate >= X<br/>over last Y seconds"| Orch
    Orch --> Build --> Train --> Eval --> Register --> Reload
    Reload --> Runtime
```

**How it works:**

1. The fraud detection service exposes a `/feedback` endpoint that accepts ground-truth labels for past predictions.
2. Each feedback item (predicted vs. actual) is recorded in a **`FailureTracker`** — a thread-safe sliding window that maintains the last `Y` seconds of outcomes.
3. After each feedback submission, the tracker computes the failure rate (false positives + false negatives) / total.
4. When the failure rate crosses the configured threshold `X` with at least `min_samples` in the window:
   - A background daemon thread triggers the **`AutoTrainingOrchestrator`**.
   - The orchestrator builds a fresh training dataset, trains a new XGBoost model with Optuna tuning, and evaluates quality against a minimum baseline.
   - If the new model passes quality checks, it is registered in MLflow and transitioned to the Production stage.
   - The BentoML service **hot-reloads** the new model by atomically swapping the global `FraudModelRuntime` singleton.
   - The failure tracker window is reset so the new model starts with a clean slate.
5. A configurable **cooldown** prevents rapid back-to-back retrains.

**Configuration** (`common/config.py` → `AutoTrainingConfig`):

| Parameter | Env Var | Default | Description |
|-----------|---------|---------|-------------|
| `enabled` | `AUTO_TRAINING_ENABLED` | `true` | Master switch |
| `failure_rate_threshold` | `AUTO_TRAINING_FAILURE_RATE_THRESHOLD` | `0.4` | Failure rate (0–1) that triggers retraining |
| `monitoring_window_seconds` | `AUTO_TRAINING_MONITORING_WINDOW_SECONDS` | `300` | Sliding window size in seconds |
| `cooldown_seconds` | `AUTO_TRAINING_COOLDOWN_SECONDS` | `120` | Minimum wait between retrains |
| `min_samples` | `AUTO_TRAINING_MIN_SAMPLES` | `20` | Minimum feedback samples before rate is meaningful |
| `training_dataset_size` | `AUTO_TRAINING_TRAINING_DATASET_SIZE` | `512` | Rows in retraining dataset |

**Key design decisions:**

- **Thread-safe sliding window** (`collections.deque` + `threading.Lock`) — bounded memory, O(1) append, lazy pruning.
- **Callback-driven** — the tracker fires a callback in a daemon thread, never blocking the prediction hot path.
- **Fail-safe** — if training or registration fails, the existing production model continues serving unchanged.
- **Atomic hot-reload** — the global `_RUNTIME` reference is swapped in a single Python assignment; in-flight requests complete with the old model, subsequent ones use the new one.

**Pseudo-code:**

```
STATE: outcomes = deque()  // sliding window of (timestamp, predicted, actual)
STATE: last_trigger_time = 0
STATE: training_in_progress = false

ON /feedback (items):
    FOR item IN items:
        outcomes.append((now(), item.predicted, item.actual))
    prune entries older than monitoring_window_seconds
    IF len(outcomes) >= min_samples:
        failures = count(o for o in outcomes if o.predicted != o.actual)
        rate = failures / len(outcomes)
        IF rate >= failure_rate_threshold
           AND NOT training_in_progress
           AND (now() - last_trigger_time) >= cooldown_seconds:
            training_in_progress = true
            last_trigger_time = now()
            SPAWN THREAD:
                dataset = build_synthetic_dataset(training_dataset_size)
                result = train_fraud_model(dataset)
                IF result.roc_auc >= 0.60 AND result.accuracy >= 0.65:
                    register_model_in_mlflow(result.run_id)
                    hot_reload_model()
                    outcomes.clear()  // fresh window for new model
                training_in_progress = false
```

---

## 7. Orchestration (Airflow)

Three DAGs live under `services/airflow/dags/`.

**feature_engineering_pipeline (hourly):**

```mermaid
flowchart LR
    Start1[start] --> Extract[extract_realtime_events]
    Extract --> Compute[compute_batch_features]
    Compute --> Validate[validate_features]
    Validate --> Materialize[materialize_to_feast]
    Materialize --> DriftCheck[check_for_drift]
    DriftCheck --> MaybeTrain[maybe_trigger_training]
    MaybeTrain --> End1[end]
```

- Extract: Kafka consumption (stubbed locally).
- Compute: Aggregates to e.g. `data/processed/feast/event_aggregates.parquet`.
- Validate: Basic data checks (Great Expectations stubbed).
- Materialize: Write to Feast.
- Drift check and maybe_trigger_training: Conditional retraining path (stub).

**model_training_pipeline (weekly):**

```mermaid
flowchart LR
    Start2[start] --> Build[build_training_dataset]
    Build --> Train[train_model]
    Train --> Eval[evaluate_against_baseline]
    Eval --> Register[register_and_mark_candidate]
    Register --> Canary[deploy_canary]
    Canary --> Monitor[monitor_canary]
    Monitor --> Promote[finalize_promotion]
    Promote --> End2[end]
```

**ml_monitoring_pipeline (daily):** See section 6 (collect → compute drift → check thresholds → send_alerts, trigger_retraining).

---

## 8. Simulator

The simulator generates realistic ticketing traffic (events, users, transactions, fraud patterns) to stress-test the platform.

**Architecture:**

```mermaid
flowchart TB
    subgraph core [Core Generators]
        EventGen[EventGenerator]
        UserGen[UserGenerator]
        TxnGen[TransactionGenerator]
        FraudGen[FraudSimulator]
    end

    subgraph scenarios [Scenarios]
        Normal[NormalTraffic]
        Flash[FlashSale]
        Fraud[FraudAttack]
        FraudDrift["FraudDriftRetrain<br/>(novel fraud → auto-retrain)"]
        Drift[GradualDrift]
        Degradation[SystemDegradation]
        BlackFriday[BlackFriday]
    end

    subgraph validators [Validators]
        LatencyV[LatencyValidator]
        AccuracyV[AccuracyValidator]
        DriftV[DriftValidator]
        BusinessV[BusinessValidator]
    end

    subgraph output [Output]
        ReportGen[ReportGenerator]
    end

    scenarios --> core
    scenarios --> validators
    scenarios --> ReportGen
```

**Scenario lifecycle:**

```
FUNCTION execute():
    setup()      // Create events, users, state
    run()        // Generate traffic / call APIs or synthetic responses
    teardown()   // Aggregate results (latency, errors, drift, etc.)
    RETURN validate()  // Compare results to expected_metrics with 10% tolerance
```

**Validation:** For each `expected_metrics` entry, actual must be within 10% of expected (or the scenario fails that metric).

**Key generators (conceptual):**

- **EventGenerator:** Categories (sports, concert, family, cultural, etc.), power-law popularity, pricing tiers (VIP/Premium/Standard), Saudi venues/dates.
- **UserGenerator:** Personas (casual, enthusiast, VIP, scalper, fraudster) with purchase_frequency, price_sensitivity, fraud_probability, bot_likelihood, cart_abandonment_rate.
- **TransactionGenerator:** num_tickets, tier, price, payment_method, device/session, fraud_indicators (velocity, device_mismatch, geo_mismatch, time_anomaly), is_fraud.
- **FraudSimulator:** Patterns (credential_stuffing, card_testing, bot_scalping) with configurable attack and success rates.

**Scenarios:**

| Scenario | Purpose |
|----------|---------|
| `normal-traffic` | Baseline operations (~3% fraud) |
| `flash-sale` | Mega-event launch (10K+ RPS) |
| `fraud-attack` | Coordinated fraud (credential stuffing, card testing, bot scalping) |
| `fraud-drift-retrain` | **Novel fraud patterns** (account takeover, synthetic identity, refund abuse) that cause model degradation and trigger auto-retraining |
| `gradual-drift` | Seasonal behavior changes |
| `system-degradation` | Partial service failures |
| `black-friday` | Extreme load |
| `mix` | Weighted combination of scenarios |

**fraud-drift-retrain scenario details:** This scenario introduces three novel fraud vectors that the production model has never seen, producing feature distributions different from the original training data. Account takeover uses high `lifetime_purchases` with low `fraud_risk_score`; synthetic identity uses zero history; refund abuse uses moderate features that are hard to distinguish from legitimate users. The model misses ~70% of these novel patterns, causing the failure rate to breach the auto-training threshold and trigger retraining.

**CLI:** `simulator.cli` — `list-scenarios`, `run <scenario> -o report.html`, `run-all -o reports/`, `realtime <scenario> --duration 60 --rps 100`.

---

## 9. Configuration System

**Location:** `common/config.py`

**Style:** Pydantic-based, environment-driven. `ENVIRONMENT=local` vs `production` selects defaults (e.g. Feast offline store: DuckDB vs BigQuery; storage: local paths vs GCS).

**Config classes (summary):**

- **PostgresConfig** — host, port, user, password, database names (airflow, mlflow).
- **RedisConfig** — host, port, password (optional).
- **MinioConfig** — endpoint, access key, secret key, bucket.
- **FeastConfig** — offline store type, DuckDB path or BigQuery dataset.
- **MlflowConfig** — tracking_uri, artifact root.
- **AirflowConfig** — webserver URL.
- **StorageConfig** — GCS bucket for production.
- **AutoTrainingConfig** — failure_rate_threshold, monitoring_window_seconds, cooldown_seconds, min_samples, training_dataset_size (see section 6.1).

Config is loaded via `get_config()` and used by feature utils, model utils, and BentoML common code so the same codebase works locally and in production with different env vars.
