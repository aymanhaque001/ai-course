# Module 15: MLOps & Production Machine Learning

> **Prerequisites:** Module 4 (Training), Module 12 (Classical ML)  
> **Estimated Time:** 10-12 hours  
> **Relevance:** Building models is <20% of production ML. This module covers the other 80%: deployment, monitoring, pipelines, and infrastructure

---

## 15.1 The Production ML Gap

```
┌──────────────────────────────────────────────────────────────────┐
│          "ML in Research" vs "ML in Production"                    │
│                                                                   │
│  Research:                    Production:                          │
│  ┌──────────┐                 ┌──────────────────────────────┐   │
│  │          │                 │    Data     │  Monitoring    │   │
│  │  MODEL   │                 │  Pipeline   │  & Alerting    │   │
│  │          │                 ├─────────────┤────────────────┤   │
│  └──────────┘                 │Feature Eng  │  A/B Testing   │   │
│                               ├─────────────┤────────────────┤   │
│  Jupyter notebook,            │   MODEL     │  Deployment    │   │
│  clean dataset,               ├─────────────┤────────────────┤   │
│  one metric                   │  Training   │  Serving Infra │   │
│                               │  Pipeline   │  & Scaling     │   │
│                               ├─────────────┤────────────────┤   │
│                               │  Eval &     │  CI/CD for ML  │   │
│                               │  Validation │  & Governance  │   │
│                               └──────────────────────────────┘   │
│                                                                   │
│  "Only a small fraction of real-world ML systems is composed      │
│   of the ML code."  — Google, "Hidden Technical Debt in ML"       │
└──────────────────────────────────────────────────────────────────┘
```

---

## 15.2 ML Pipeline Architecture

### End-to-End ML Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    Production ML Pipeline                          │
│                                                                   │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │  Data   │──▶│ Feature  │──▶│ Training │──▶│  Model   │      │
│  │Ingestion│   │Engineering│   │          │   │Validation│      │
│  └─────────┘   └──────────┘   └──────────┘   └────┬─────┘      │
│       │                                            │             │
│       │         ┌──────────────────────────────────┘             │
│       │         │                                                │
│       │         ▼                                                │
│       │    ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│       │    │  Model   │──▶│  Deploy  │──▶│ Monitor  │           │
│       │    │ Registry │   │ (Serve)  │   │ & Alert  │           │
│       │    └──────────┘   └──────────┘   └────┬─────┘           │
│       │                                        │                 │
│       └────────────────────────────────────────┘                 │
│                    Feedback Loop                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Pipeline Orchestration Tools

```
┌──────────────┬──────────────────────────────────────────────────┐
│  Airflow     │ General-purpose workflow orchestrator             │
│              │ DAG-based, Python, mature ecosystem               │
│              │ Best for: data pipelines, ETL + ML                │
├──────────────┼──────────────────────────────────────────────────┤
│  Kubeflow    │ ML-native on Kubernetes                           │
│  Pipelines   │ Components = containers, supports GPU             │
│              │ Best for: K8s-native ML workflows                 │
├──────────────┼──────────────────────────────────────────────────┤
│  Prefect     │ Modern Python workflow engine                     │
│              │ Dynamic workflows, better error handling          │
│              │ Best for: Python-native teams                     │
├──────────────┼──────────────────────────────────────────────────┤
│  Dagster     │ Data-aware orchestrator                           │
│              │ Software-defined assets, strong typing            │
│              │ Best for: data engineering + ML integration       │
├──────────────┼──────────────────────────────────────────────────┤
│  ZenML       │ MLOps framework                                   │
│              │ Pipeline abstraction, integrates with all infra   │
│              │ Best for: standardizing ML workflows              │
└──────────────┴──────────────────────────────────────────────────┘
```

---

## 15.3 Experiment Tracking

```
Why Track Experiments?

  Run #142: lr=3e-4, batch=32, epochs=5  → F1=0.847
  Run #143: lr=1e-4, batch=64, epochs=3  → F1=0.862  ← best!
  Run #144: lr=1e-3, batch=32, epochs=10 → F1=0.823

  Without tracking: "Which config worked best? 🤷"
  With tracking: Compare N runs across all hyperparams, metrics, artifacts

┌──────────────────────────────────────────────────────────────────┐
│              Experiment Tracking Tools                             │
├──────────────┬──────────────────────────────────────────────────┤
│  Weights &   │ Most popular. Cloud-hosted or self-hosted.        │
│  Biases      │ Log metrics, hyperparams, artifacts, system stats │
│  (W&B)       │ Beautiful dashboards, sweeps, reports             │
│              │ LLM-specific: trace prompts, token usage          │
├──────────────┼──────────────────────────────────────────────────┤
│  MLflow      │ Open-source, self-hosted. Model registry.         │
│              │ Tracking, projects, models, deployment            │
│              │ Databricks-backed, strong enterprise adoption     │
├──────────────┼──────────────────────────────────────────────────┤
│  Neptune     │ Cloud-hosted, great for team collaboration        │
│              │ Strong metadata management                        │
├──────────────┼──────────────────────────────────────────────────┤
│  CometML     │ Similar to W&B, good visualization               │
│              │ LLM experiment tracking support                   │
├──────────────┼──────────────────────────────────────────────────┤
│  TensorBoard │ Free, built into TensorFlow/PyTorch              │
│              │ Basic but sufficient for individual researchers   │
└──────────────┴──────────────────────────────────────────────────┘

What to Track:
  ┌─────────────────────────────────────────────────────────┐
  │  Hyperparameters:  lr, batch_size, model_config, seed   │
  │  Metrics:          loss, accuracy, F1, BLEU, perplexity │
  │  Artifacts:        model weights, configs, data samples │
  │  System:           GPU util, memory, training time      │
  │  Code:             Git commit hash, diff                │
  │  Data:             Dataset version, preprocessing steps │
  │  Environment:      Python version, package versions     │
  └─────────────────────────────────────────────────────────┘
```

---

## 15.4 Data Management for ML

### Data Versioning

```
Code has Git. Data needs versioning too.

DVC (Data Version Control):

  myproject/
  ├── data/
  │   ├── train.csv        ← actual data (large, NOT in git)
  │   └── train.csv.dvc    ← metadata pointer (small, IN git)
  ├── models/
  │   ├── model.pkl        ← trained model (large, NOT in git)
  │   └── model.pkl.dvc    ← metadata pointer (small, IN git)
  └── dvc.yaml             ← pipeline definition

  $ dvc add data/train.csv        # Track data file
  $ git add data/train.csv.dvc    # Commit metadata
  $ dvc push                      # Push data to remote storage (S3)

  $ git checkout v1.0 && dvc checkout  # Restore data from any commit!

Data Storage Options:
  S3, GCS, Azure Blob, NFS, local
  DVC handles the mapping: git commit → data version
```

### Data Versioning — Schema Evolution

```
When data schemas change (columns added/removed, types changed), models
trained on the old schema can break silently in production.

Common schema change scenarios:
  - New feature column added to upstream data warehouse
  - Column renamed (user_id → customer_id)
  - Type change (string "1.5" → float 1.5, or int → nullable int)
  - Feature removed (deprecated data source shut down)
  - New category value added to a categorical feature (unseen by model)

Solutions:
  (1) Schema Registry (e.g., Confluent Schema Registry, AWS Glue Schema Registry)
        Enforce backward compatibility: new schemas must be readable by
        models expecting the old schema (additive changes only).
        Breaking changes require a new schema version and an explicit migration.

  (2) Feature Deprecation Policy
        Mark old features as deprecated → alert all dependent models/teams
        → allow a migration period (e.g., 30 days)
        → confirm no active models read the feature → then remove.
        Never silently drop features from a live pipeline.

  (3) Schema Validation in CI
        Read the model's expected input schema from the registry.
        On every PR that touches a data pipeline:
          - Assert the new pipeline output matches the registered schema.
          - Fail CI if any required field is missing, renamed, or type-changed.
        Catches silent breakage before it reaches production.

  (4) Feature Versioning
        Decouple schema evolution from model update cycles:
          user_spend_v1  →  original computation (still active)
          user_spend_v2  →  new computation logic (gradual adoption)
        Models explicitly declare which version they depend on.
        Multiple versions coexist in the feature store during migration.
        Each consumer migrates on its own timeline without forced coordination.
```

### Data Quality

```
┌──────────────────────────────────────────────────────────────────┐
│                    Data Quality Framework                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  SCHEMA VALIDATION                                                │
│    Expected columns, types, ranges                                │
│    Tools: Great Expectations, Pandera, Pydantic                   │
│                                                                   │
│  STATISTICAL CHECKS                                               │
│    Distribution comparisons (PSI, KS test)                        │
│    Missing value rates, cardinality changes                       │
│    Outlier detection (IQR, z-score)                               │
│                                                                   │
│  ML-SPECIFIC CHECKS                                               │
│    Label distribution consistency                                 │
│    Feature-target correlations stable?                             │
│    Data leakage detection (future data in training)               │
│    Class balance monitoring                                       │
│                                                                   │
│  FRESHNESS                                                        │
│    Is data pipeline running on schedule?                          │
│    How old is the most recent record?                             │
│    SLA: data available within X hours of collection               │
│                                                                   │
│  For LLM Training Data:                                           │
│    Deduplication (MinHash, exact match)                            │
│    PII detection and removal                                      │
│    Toxicity/bias scanning                                         │
│    Benchmark contamination checks                                 │
│    License compliance verification                                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Feature Stores

```
Feature Store Architecture:

  ┌───────────────────────────────────────────────────────────┐
  │                    Feature Store                          │
  │                                                           │
  │  ┌─────────────┐        ┌──────────────┐                 │
  │  │  Offline    │        │   Online     │                 │
  │  │  Store      │        │   Store      │                 │
  │  │  (Batch)    │        │   (Real-time)│                 │
  │  │             │  sync  │              │                 │
  │  │  Data Lake  │───────▶│   Redis /    │                 │
  │  │  (Parquet)  │        │   DynamoDB   │                 │
  │  │             │        │              │                 │
  │  │  For:       │        │  For:        │                 │
  │  │  Training   │        │  Inference   │                 │
  │  │  Batch jobs │        │  Real-time   │                 │
  │  └─────────────┘        └──────────────┘                 │
  │                                                           │
  │  Feature definitions: user_avg_spend_7d, item_popularity │
  │  Versioned, documented, shared across teams              │
  │                                                           │
  │  Tools: Feast (open source), Tecton, Hopsworks            │
  └───────────────────────────────────────────────────────────┘

Why Feature Stores Matter:
  - Training-serving skew: ensure features are computed
    identically during training and inference
  - Reusability: same features shared across models/teams
  - Point-in-time correctness: avoid data leakage
  - Low latency: pre-computed features for real-time serving
  - Offline/Online consistency: same transformation code used for both stores
```

### Offline/Online Store Consistency

```
The dual-store pattern creates a critical consistency challenge:

  Offline store: batch jobs (Spark) compute features; results may be hours old.
                 Used for: model training, batch scoring pipelines.
  Online store:  pre-computed features served in <5 ms (Redis/DynamoDB).
                 Used for: real-time inference at request time.

  Risk: if the two pipelines compute the same feature with slightly different
        logic, the model trains on data it never sees in serving → silent failure.

  Point-in-time correctness (offline training data):
    The offline pipeline must compute features as they *would have been* at
    training time — using no data from after the label cutoff (data leakage).
    e.g., "user_purchases_7d" for a Jan 1 training example must count only
    purchases before Jan 1, not purchases that happened after.

  Solutions:
    (1) Single transformation library: same function executes online (Python
        at request time) and offline (PySpark batch), eliminating dual-
        implementation drift at the source.
    (2) Backfill: when online feature logic changes, re-run the batch pipeline
        to backfill the offline store before triggering a model retrain.
    (3) Integration tests: compute features for N sample entity keys both online
        (Redis lookup) and offline (batch recompute); assert values match within
        an acceptable tolerance on every deployment.
```

### Training-Serving Skew

```
Training-serving skew is the #1 cause of silent ML failures in production.

The model performs well in offline evaluation but unexpectedly poorly in
production because the features at inference time differ from training time.

Causes:
  (1) Feature computation differences
        Train:  pandas one-liner on a DataFrame column
        Serve:  different SQL query — same intent, subtly different result
                (e.g., NULL handling, rounding, time zone offset)
  (2) Data preprocessing mismatches
        Train:  StandardScaler.fit_transform(X_train)  ← fits on training set
        Serve:  raw feature passed directly to model   ← scaler never loaded!
  (3) Library version differences
        scikit-learn 1.1 → 1.3 changed default hyperparameters;
        numpy dtype handling changed across versions; silent numeric drift.
  (4) Stale features (feature store TTL expired)
        Feature store key expires; model receives fallback value (0 or null).
        Model trained only on fresh features → distribution shifts silently.
  (5) Schema mismatches between training and serving pipelines
        New feature column added to training; serving pipeline not updated.

Detection:
  - Log feature vectors at both training time and inference time.
  - Compare feature distributions between training logs and live serving
    logs using PSI or KS test on a sampled window of production traffic.
  - Shadow mode: run new pipeline on live traffic and compare feature
    statistics to the production pipeline before any go-live.
  - Integration test: given identical raw input, assert training pipeline
    and serving pipeline produce identical feature vectors.

Prevention:
  - Feature store (Feast/Tecton): one computation, served identically to
    both training jobs and real-time inference.
  - Serialize ALL preprocessing objects (scalers, encoders, imputers)
    as model artifacts; load them at serve time — never refit on production.
  - Pin library versions in both training and serving containers (same image).
  - Add automated feature-value comparison tests to CI on every deploy.
```

---

## 15.5 Model Serving & Deployment

### Deployment Patterns

```
┌──────────────┬──────────────────────────────────────────────────┐
│ Batch        │ Run predictions on a schedule (hourly/daily)     │
│ Inference    │ e.g., product recommendations, risk scores       │
│              │ Store results in database, serve from cache      │
│              │ Simplest pattern. Use when latency < 1hr is OK.  │
├──────────────┼──────────────────────────────────────────────────┤
│ Real-time    │ Synchronous API call → prediction                │
│ Inference    │ REST/gRPC endpoint, <100ms latency               │
│              │ e.g., fraud detection, search ranking             │
│              │ Requires model serving infrastructure            │
├──────────────┼──────────────────────────────────────────────────┤
│ Streaming    │ Process events as they arrive (Kafka/Kinesis)    │
│ Inference    │ Near-real-time, event-driven                     │
│              │ e.g., anomaly detection, real-time personalization│
├──────────────┼──────────────────────────────────────────────────┤
│ Edge/On-     │ Run model on user's device                       │
│ Device       │ No network latency, works offline                │
│              │ e.g., Siri, keyboard prediction, camera filters  │
│              │ Requires: quantization, small models, ONNX       │
├──────────────┼──────────────────────────────────────────────────┤
│ LLM-specific │ Streaming responses (token-by-token SSE)         │
│              │ Long-running inference (seconds, not ms)          │
│              │ High GPU memory, KV cache management              │
│              │ vLLM, TGI, TensorRT-LLM                          │
└──────────────┴──────────────────────────────────────────────────┘
```

### Model Serving Infrastructure

```
┌──────────────────────────────────────────────────────────────────┐
│                Model Serving Stack                                │
│                                                                   │
│  Layer 1: MODEL FORMAT                                            │
│    PyTorch (.pt) → TorchScript → ONNX → TensorRT                │
│    Optimization: graph fusion, kernel auto-tuning, quantization  │
│                                                                   │
│  Layer 2: SERVING FRAMEWORK                                       │
│  ┌──────────────┬────────────────────────────────────────┐       │
│  │ TorchServe   │ PyTorch native, multi-model, batching  │       │
│  │ Triton       │ NVIDIA, multi-framework, high perf     │       │
│  │ TF Serving   │ TensorFlow native, production-proven   │       │
│  │ vLLM         │ LLM-optimized (PagedAttention)         │       │
│  │ TGI          │ HuggingFace LLM serving                │       │
│  │ BentoML      │ Python-native, easy packaging          │       │
│  │ Ray Serve    │ Distributed, multi-model pipelines     │       │
│  └──────────────┴────────────────────────────────────────┘       │
│                                                                   │
│  Layer 3: INFRASTRUCTURE                                          │
│    Kubernetes + GPU operator                                      │
│    Auto-scaling (HPA based on GPU utilization or queue depth)     │
│    Load balancing (round-robin or least-connections)              │
│                                                                   │
│  Layer 4: API GATEWAY                                             │
│    Rate limiting, authentication, request routing                 │
│    A/B traffic splitting, canary deployments                      │
│    API versioning (model version → API version)                   │
└──────────────────────────────────────────────────────────────────┘
```

### Containerization & Packaging

```
Model Packaging:

  Dockerfile for ML Model:
  ┌──────────────────────────────────────────────┐
  │  FROM python:3.11-slim                       │
  │  COPY requirements.txt .                     │
  │  RUN pip install -r requirements.txt         │
  │  COPY model/ /app/model/                     │
  │  COPY serve.py /app/                         │
  │  EXPOSE 8080                                 │
  │  CMD ["python", "/app/serve.py"]             │
  └──────────────────────────────────────────────┘

  For GPU:
  ┌──────────────────────────────────────────────┐
  │  FROM nvidia/cuda:12.1-runtime-ubuntu22.04   │
  │  # ... install PyTorch with CUDA             │
  └──────────────────────────────────────────────┘

Model Registry (MLflow / W&B):
  ┌──────────────────────────────────────────────────┐
  │  Model: fraud-detector                           │
  │  ├── v1 (staging)  - XGBoost, F1=0.92           │
  │  ├── v2 (production) - LightGBM, F1=0.95        │
  │  └── v3 (testing)  - Transformer, F1=0.96       │
  │                                                   │
  │  Promotion: testing → staging → production        │
  │  Rollback: production → previous version          │
  │  Lineage: which data, code, and config produced  │
  │           this model?                             │
  └──────────────────────────────────────────────────┘
```

---

## 15.6 CI/CD for Machine Learning

```
┌──────────────────────────────────────────────────────────────────┐
│            CI/CD Pipeline for ML                                  │
│                                                                   │
│  Code Change (PR)                                                 │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────────────────────────────────────────┐                 │
│  │  CI: Continuous Integration                  │                 │
│  │                                              │                 │
│  │  1. Lint & type check (ruff, mypy)          │                 │
│  │  2. Unit tests (model logic, data transforms)│                │
│  │  3. Data validation (schema checks)          │                 │
│  │  4. Small-scale training test (1 epoch, tiny)│                │
│  │  5. Model quality gate (eval on holdout)     │                │
│  │  6. Integration tests (API endpoints)        │                 │
│  └─────────────────────┬───────────────────────┘                 │
│                        │ pass?                                    │
│                        ▼                                          │
│  ┌─────────────────────────────────────────────┐                 │
│  │  CD: Continuous Deployment                   │                 │
│  │                                              │                 │
│  │  1. Build container image                    │                 │
│  │  2. Push to registry (ECR, GCR)              │                 │
│  │  3. Deploy to staging environment            │                 │
│  │  4. Run integration tests on staging         │                 │
│  │  5. Canary deployment (5% traffic)           │                 │
│  │  6. Monitor metrics for 1 hour               │                 │
│  │  7. Progressive rollout (25% → 50% → 100%)  │                 │
│  │  8. Automated rollback if metrics degrade    │                 │
│  └─────────────────────────────────────────────┘                 │
│                                                                   │
│  Tools: GitHub Actions, GitLab CI, Jenkins, Argo Workflows       │
│                                                                   │
│  ML-specific additions to standard CI/CD:                        │
│  - Model quality gates (metrics must improve or match)            │
│  - Data validation step                                           │
│  - Model size & latency checks                                    │
│  - Shadow mode testing (run new model alongside old, compare)    │
│  - Automatic retraining triggers (data drift detected)           │
└──────────────────────────────────────────────────────────────────┘
```

### Retraining Triggers — Code Changes vs Data Changes

```
Two fundamentally different situations require very different responses:

  CODE CHANGE (bug fix, new model architecture, dependency update)
  → Standard CI/CD pipeline — NO retraining needed:
      lint → unit test → integration test → build container → deploy
      Model weights are unchanged; the fix is in code, not learned parameters.

  DATA CHANGE or DRIFT (distribution shift, new patterns, concept drift)
  → Trigger the retraining pipeline:
      detect drift → pull fresh training data → re-run training job
      → evaluate new model → compare against current production model
      → promote if new model wins on holdout eval; keep current if not.

Trigger types:
  ┌─────────────────┬──────────────────────────────────────────────┐
  │ Schedule-based  │ Retrain weekly/monthly as a safety net.       │
  │                 │ Catches slow drift even if no alert fires.    │
  ├─────────────────┼──────────────────────────────────────────────┤
  │ Event-based     │ Drift detector fires (PSI > 0.25, KS p<0.05).│
  │                 │ Business metric drops beyond alert threshold. │
  │                 │ Immediately triggers the retraining pipeline. │
  ├─────────────────┼──────────────────────────────────────────────┤
  │ Manual          │ Domain expert confirms distribution changed   │
  │                 │ (product launch, regulatory change, new       │
  │                 │  market segment onboarded).                   │
  └─────────────────┴──────────────────────────────────────────────┘

Promotion gate (data-triggered retraining):
  The retrained model must outperform the production model on:
    - Offline holdout set (same evaluation metrics used during training)
    - Shadow mode on live traffic (compare outputs against production model)
  If new model wins → canary → progressive rollout → full promotion.
  If new model is worse → do NOT promote; investigate data quality issues.

Clean pipeline routing:
    git push (code change)   → code CI/CD branch   → deploy (no retrain)
    drift alert (data change) → retraining branch  → eval → conditional deploy
```

---

## 15.7 Monitoring & Observability

### What to Monitor

```
┌──────────────────────────────────────────────────────────────────┐
│                 ML Monitoring Dashboard                            │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  SYSTEM METRICS (same as any service):                            │
│    Latency:    P50, P95, P99 response time                       │
│    Throughput: Requests per second                                │
│    Errors:     Error rate, HTTP 5xx rate                          │
│    Resources:  CPU/GPU utilization, memory, disk                  │
│                                                                   │
│  MODEL PERFORMANCE METRICS (ML-specific):                         │
│    Prediction distribution: Has the output distribution shifted?  │
│    Confidence scores: Are predictions becoming less confident?    │
│    Business metrics: CTR, conversion, revenue per prediction     │
│    Error analysis: What types of errors are increasing?           │
│                                                                   │
│  DATA/INPUT METRICS (ML-specific):                               │
│    Feature distributions: Have inputs changed? (drift)            │
│    Missing values: Is data pipeline broken?                       │
│    Volume: Expected number of predictions happening?              │
│    Outliers: Unusual input patterns?                              │
│                                                                   │
│  LLM-SPECIFIC METRICS:                                           │
│    Token usage: Input/output tokens per request                  │
│    Hallucination rate: Factual accuracy checks                   │
│    Refusal rate: How often does the model refuse to answer?      │
│    Safety flags: Content moderation trigger rate                 │
│    RAG retrieval quality: Relevance scores trending down?         │
│    Cost per query: Track spending across model providers          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Data Drift Detection

```
Data Drift: Input data distribution changes over time

Training data:                  Production data (6 months later):
  age: mean=35, std=10            age: mean=28, std=8    ← DRIFT!
  income: mean=$50K               income: mean=$65K      ← DRIFT!

Types of Drift:
  ┌──────────────┬────────────────────────────────────────────┐
  │ Data drift   │ P(X) changes. Input distribution shifts.   │
  │ (covariate)  │ e.g., new user demographics                │
  ├──────────────┼────────────────────────────────────────────┤
  │ Concept      │ P(Y|X) changes. Same inputs, different     │
  │ drift        │ correct outputs. e.g., user preferences    │
  │              │ evolve, market conditions change            │
  ├──────────────┼────────────────────────────────────────────┤
  │ Label drift  │ P(Y) changes. Target distribution shifts.  │
  │              │ e.g., fraud rate increases seasonally       │
  └──────────────┴────────────────────────────────────────────┘

Detection Methods:
  PSI (Population Stability Index):
    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    PSI < 0.1:  No significant drift
    PSI 0.1-0.25: Moderate drift (investigate)
    PSI > 0.25: Severe drift (retrain!)

  Worked Numerical Example (PSI):
    Training distribution bin %:   [0.20, 0.30, 0.30, 0.20]
    Production distribution bin %: [0.25, 0.35, 0.25, 0.15]

    PSI = Σ (A_i − E_i) · ln(A_i / E_i)
        = (0.25−0.20)·ln(0.25/0.20) + (0.35−0.30)·ln(0.35/0.30)
        + (0.25−0.30)·ln(0.25/0.30) + (0.15−0.20)·ln(0.15/0.20)
        = (0.05)·ln(1.25)  + (0.05)·ln(1.167)
        + (−0.05)·ln(0.833) + (−0.05)·ln(0.75)
        ≈ (0.05)(0.223) + (0.05)(0.154) + (0.05)(0.182) + (0.05)(0.288)
        ≈  0.011 + 0.008 + 0.009 + 0.014
        ≈  0.042  → PSI < 0.1: No significant drift ✓

  Drifted example (PSI > 0.25):
    Training:   [0.20, 0.30, 0.30, 0.20]
    Production: [0.05, 0.55, 0.10, 0.30]  ← large population shift
    PSI ≈ 0.31  → PSI > 0.25: Severe drift — RETRAIN!

  KS Test (Kolmogorov-Smirnov):
    Statistical test for distribution difference
    p < 0.05 → significant drift

  Monitoring Tools: Evidently AI, WhyLabs, Arize, NannyML
```

### Alerting Strategy

```
Alert Severity Levels:

  ┌────────────┬───────────────────────────────────────────────┐
  │ CRITICAL   │ Model serving is DOWN or returning errors     │
  │ (page)     │ Latency > 10× normal                         │
  │            │ Safety filter triggering at 5× normal rate    │
  ├────────────┼───────────────────────────────────────────────┤
  │ HIGH       │ Model accuracy dropped > 5%                   │
  │ (Slack)    │ Data pipeline delayed > 2 hours               │
  │            │ Severe data drift detected (PSI > 0.25)       │
  ├────────────┼───────────────────────────────────────────────┤
  │ MEDIUM     │ Moderate drift detected (PSI 0.1-0.25)        │
  │ (ticket)   │ Prediction volume anomaly (±30%)              │
  │            │ GPU utilization consistently > 90%             │
  ├────────────┼───────────────────────────────────────────────┤
  │ LOW        │ Minor metric fluctuations                     │
  │ (log)      │ Upcoming model expiration (retrain due)       │
  └────────────┴───────────────────────────────────────────────┘

Anti-patterns:
  ✗ Alert on every metric fluctuation (alert fatigue)
  ✗ No alerting at all (problems go unnoticed)
  ✗ Same severity for all alerts
  ✓ Alert on TRENDS, not individual data points
  ✓ Include runbook links in alerts
  ✓ Auto-remediation where possible (retrain, rollback)
```

---

## 15.8 A/B Testing for ML Models

```
┌──────────────────────────────────────────────────────────────────┐
│             A/B Testing ML Models                                 │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  SETUP                                                            │
│    Control (A): Current production model                          │
│    Treatment (B): New model candidate                             │
│    Splitting: Hash(user_id) % 100 < 50 → control                │
│    Duration: 1-4 weeks (depending on traffic volume)              │
│                                                                   │
│  METRICS FRAMEWORK                                                │
│    Primary: The metric you're optimizing                          │
│      e.g., click-through rate, conversion, revenue                │
│    Secondary: Related metrics you expect to improve               │
│      e.g., engagement time, return rate                           │
│    Guardrail: Metrics that MUST NOT degrade                       │
│      e.g., latency, error rate, user complaints                   │
│                                                                   │
│  STATISTICAL RIGOR                                                │
│    Sample size: Calculate BEFORE starting (power analysis)        │
│    Significance: p < 0.05 AND practical significance (MDE)        │
│    Multiple comparisons: Bonferroni correction                    │
│    No peeking: Don't check results daily and stop early           │
│    (or use sequential testing / always-valid p-values)            │
│                                                                   │
│  DEPLOYMENT PATTERNS                                              │
│    Shadow mode:   New model runs alongside, predictions logged    │
│                   but NOT shown to users (validate before expose) │
│    Canary:        5% traffic → 25% → 50% → 100%                  │
│    Interleaving:  Mix results from both models in same response   │
│                   (common for search/recommendation)              │
│                                                                   │
│  LLM-SPECIFIC A/B TESTING                                        │
│    Challenge: Long-form output, hard to measure "quality"         │
│    Approach: LLM-as-judge + human eval sample + business metrics  │
│    Metric: User thumbs-up rate, task completion, follow-up rate   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 15.9 LLM-Specific MLOps

### LLMOps — What's Different

```
┌──────────────────────────────────────────────────────────────────┐
│              Traditional MLOps vs LLMOps                          │
├────────────────┬─────────────────┬───────────────────────────────┤
│                │ Traditional ML   │ LLMOps                       │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Model          │ Train from       │ Prompt/fine-tune a           │
│                │ scratch          │ foundation model             │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Data           │ Structured,      │ Unstructured text, prompts,  │
│                │ tabular          │ preference pairs             │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Versioning     │ Model weights,   │ Prompts, system messages,    │
│                │ code, data       │ RAG configs, guardrails      │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Evaluation     │ Standard metrics │ LLM-as-judge, human eval,    │
│                │ (F1, AUC)        │ task-specific evals           │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Monitoring     │ Prediction drift │ Hallucination, safety,       │
│                │                  │ cost per query, token usage  │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Cost           │ Training-heavy   │ Inference-heavy              │
│                │                  │ (API calls add up!)          │
├────────────────┼─────────────────┼───────────────────────────────┤
│ Latency        │ <100ms typical   │ 1-30s typical (generation)   │
│                │                  │ Streaming mitigates TTFB     │
└────────────────┴─────────────────┴───────────────────────────────┘
```

### Prompt Management

```
Prompt Versioning & Management:

  prompts/
  ├── v1/
  │   ├── system_prompt.txt
  │   ├── few_shot_examples.json
  │   └── config.yaml  (model, temperature, max_tokens)
  ├── v2/
  │   ├── system_prompt.txt  (improved instructions)
  │   ├── few_shot_examples.json  (better examples)
  │   └── config.yaml
  └── evaluation/
      ├── test_cases.json
      └── eval_results/
          ├── v1_results.json  (accuracy: 82%)
          └── v2_results.json  (accuracy: 89%)

Tools: LangSmith, Humanloop, PromptLayer, Braintrust
Approach:
  1. Version prompts in git (like code)
  2. Run eval suite on each prompt change
  3. Deploy winning prompt through CI/CD
  4. Monitor performance in production
  5. Log all prompt-response pairs for debugging
```

### LLM Cost Optimization

```
┌──────────────────────────────────────────────────────────────────┐
│              LLM Cost Optimization Strategies                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. MODEL SELECTION                                               │
│     Not every task needs GPT-4. Route by complexity:              │
│     Simple query → small/fast model ($0.001/request)              │
│     Complex reasoning → large model ($0.05/request)               │
│     Classification → fine-tuned small model (cheapest)            │
│                                                                   │
│  2. CACHING                                                       │
│     Exact match cache: same prompt → cached response              │
│     Semantic cache: similar prompts → cached response             │
│     Embedding similarity threshold for cache hits                 │
│     Can reduce costs by 30-60% for repetitive queries             │
│                                                                   │
│  3. PROMPT OPTIMIZATION                                           │
│     Shorter prompts = fewer input tokens = lower cost             │
│     Replace verbose few-shot examples with fine-tuned model       │
│     Use prompt compression (LLMLingua)                            │
│                                                                   │
│  4. BATCH PROCESSING                                              │
│     Use batch APIs (50% cheaper on OpenAI)                        │
│     Accumulate non-urgent requests, process in batch              │
│                                                                   │
│  5. SELF-HOSTING                                                  │
│     Break-even: ~$500K+/year in API costs → consider self-host   │
│     Open-source models: LLaMA, Mistral, Qwen                     │
│     Serving: vLLM on H100 GPUs                                   │
│     Cost: ~$2-3/hour per GPU × utilization                       │
│                                                                   │
│  6. FINE-TUNING                                                   │
│     Fine-tuned small model can match large model on your task    │
│     GPT-4 → fine-tuned GPT-4o-mini → 20× cost reduction         │
│     Distillation: large model generates training data for small  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 15.10 Infrastructure & GPU Management

```
GPU Selection Guide:

┌──────────────┬─────────┬────────┬────────────────────────────┐
│ GPU          │ Memory  │ $/hour │ Best For                   │
├──────────────┼─────────┼────────┼────────────────────────────┤
│ T4           │ 16 GB   │ ~$0.5  │ Inference (small models)   │
│ A10G         │ 24 GB   │ ~$1.0  │ Inference (medium models)  │
│ L4           │ 24 GB   │ ~$0.8  │ Inference (efficient)      │
│ A100 40GB    │ 40 GB   │ ~$3.5  │ Fine-tuning, 7B inference  │
│ A100 80GB    │ 80 GB   │ ~$5.0  │ Fine-tuning, 13B inference │
│ H100         │ 80 GB   │ ~$8.0  │ Training, 70B inference    │
│ H200         │ 141 GB  │ ~$12   │ Large model training       │
│ B200         │ 192 GB  │ ~$15   │ Next-gen training          │
├──────────────┴─────────┴────────┴────────────────────────────┤
│                                                               │
│  Rule of thumb for model size & GPU memory:                   │
│    FP16: model needs ~2 × params bytes                        │
│    INT8: model needs ~1 × params bytes                        │
│    INT4: model needs ~0.5 × params bytes                      │
│    + KV cache overhead for inference                          │
│                                                               │
│  7B model FP16 → ~14GB + KV cache → A10G (24GB)              │
│  70B model INT4 → ~35GB + KV cache → A100 80GB               │
│  405B model INT4 → ~200GB → 4× A100 80GB (tensor parallel)   │
│                                                               │
└───────────────────────────────────────────────────────────────┘

Cloud Options:
  AWS:    SageMaker, EC2 P5 (H100), Inf2 (Inferentia)
  GCP:    Vertex AI, A3 (H100), TPU v5
  Azure:  Azure ML, ND H100
  Lambda: Lambda Cloud (cheaper H100s)
  Modal:  Serverless GPU (pay per second)
  Together, Replicate: Managed LLM inference APIs
```

---

## 15.11 Practical Implementation

<details>
<summary><strong>Complete Code: Production ML Pipeline Components</strong></summary>

```python
import json
import time
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

# ============================================================
# EXPERIMENT TRACKER (simplified)
# ============================================================

class ExperimentTracker:
    """Minimal experiment tracker — understand the concept before using W&B."""

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.runs = []
        self.current_run = None

    def start_run(self, run_name, hyperparams):
        self.current_run = {
            'name': run_name,
            'hyperparams': hyperparams,
            'metrics': defaultdict(list),
            'start_time': datetime.now().isoformat(),
            'git_hash': self._get_git_hash(),
        }

    def log_metric(self, name, value, step=None):
        self.current_run['metrics'][name].append({
            'value': value, 'step': step, 'time': time.time()
        })

    def end_run(self):
        self.current_run['end_time'] = datetime.now().isoformat()
        self.runs.append(self.current_run)
        self.current_run = None

    def compare_runs(self, metric_name):
        """Compare all runs by a specific metric (final value)."""
        results = []
        for run in self.runs:
            if metric_name in run['metrics']:
                final_value = run['metrics'][metric_name][-1]['value']
                results.append((run['name'], final_value, run['hyperparams']))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _get_git_hash(self):
        try:
            import subprocess
            return subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode().strip()[:8]
        except Exception:
            return 'unknown'

# ============================================================
# DATA DRIFT DETECTOR
# ============================================================

class DriftDetector:
    """Detect distribution shift between training and production data."""

    def __init__(self, reference_data, feature_names=None):
        self.reference = reference_data
        self.feature_names = feature_names or [
            f'feature_{i}' for i in range(reference_data.shape[1])
        ]

    def psi(self, expected, actual, bins=10):
        """Population Stability Index."""
        expected_percents = np.histogram(expected, bins=bins)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=bins)[0] / len(actual)

        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 0.001, None)
        actual_percents = np.clip(actual_percents, 0.001, None)

        psi_value = np.sum(
            (actual_percents - expected_percents) *
            np.log(actual_percents / expected_percents)
        )
        return psi_value

    def check_drift(self, production_data, threshold=0.25):
        """Check all features for drift."""
        alerts = []
        for i, name in enumerate(self.feature_names):
            psi_val = self.psi(self.reference[:, i], production_data[:, i])
            status = 'OK' if psi_val < 0.1 else \
                     'WARNING' if psi_val < threshold else 'ALERT'
            alerts.append({
                'feature': name,
                'psi': round(psi_val, 4),
                'status': status,
            })
        return alerts

# ============================================================
# MODEL SERVING WITH CACHING
# ============================================================

class ModelServer:
    """Production model server with caching and monitoring."""

    def __init__(self, model, cache_size=10000):
        self.model = model
        self.cache = {}
        self.cache_size = cache_size
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'latencies': [],
            'errors': 0,
        }

    def predict(self, input_data):
        """Predict with caching and monitoring."""
        self.metrics['total_requests'] += 1
        start_time = time.time()

        try:
            # Check cache
            cache_key = self._hash_input(input_data)
            if cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                return self.cache[cache_key]

            # Run model
            prediction = self.model.predict(input_data)

            # Update cache
            if len(self.cache) < self.cache_size:
                self.cache[cache_key] = prediction

            latency = time.time() - start_time
            self.metrics['latencies'].append(latency)

            return prediction

        except Exception as e:
            self.metrics['errors'] += 1
            raise

    def get_metrics(self):
        """Return serving metrics."""
        latencies = self.metrics['latencies']
        return {
            'total_requests': self.metrics['total_requests'],
            'cache_hit_rate': self.metrics['cache_hits'] /
                             max(1, self.metrics['total_requests']),
            'error_rate': self.metrics['errors'] /
                         max(1, self.metrics['total_requests']),
            'p50_latency': np.percentile(latencies, 50) if latencies else 0,
            'p99_latency': np.percentile(latencies, 99) if latencies else 0,
        }

    def _hash_input(self, data):
        return hashlib.md5(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

# ============================================================
# CANARY DEPLOYMENT MANAGER
# ============================================================

class CanaryDeployment:
    """Gradually shift traffic from old to new model."""

    def __init__(self, old_model, new_model):
        self.old_model = old_model
        self.new_model = new_model
        self.traffic_percent = 0  # % to new model
        self.old_metrics = defaultdict(list)
        self.new_metrics = defaultdict(list)

    def set_traffic(self, percent):
        """Set percentage of traffic going to new model."""
        self.traffic_percent = percent
        print(f"Traffic split: {100-percent}% old / {percent}% new")

    def route_and_predict(self, input_data, user_id):
        """Route request based on traffic split."""
        # Consistent routing per user
        use_new = (hash(user_id) % 100) < self.traffic_percent

        if use_new:
            result = self.new_model.predict(input_data)
            self.new_metrics['predictions'].append(result)
            return result, 'new'
        else:
            result = self.old_model.predict(input_data)
            self.old_metrics['predictions'].append(result)
            return result, 'old'

    def evaluate_canary(self):
        """Compare old vs new model performance."""
        # In practice, compare business metrics
        return {
            'old_model_requests': len(self.old_metrics['predictions']),
            'new_model_requests': len(self.new_metrics['predictions']),
            'traffic_percent_new': self.traffic_percent,
        }
```

</details>

---

## 15.12 Interview Questions

### Conceptual Questions

**Q1: What is training-serving skew and how do you prevent it?**

Training-serving skew occurs when the feature computation differs between training and inference, causing degraded model performance. Common causes: (1) different code paths for feature engineering in training vs serving, (2) time-dependent features computed differently (e.g., "user purchases in last 7 days" calculated at different granularity), (3) data pipeline differences (batch vs streaming). Prevention: use a feature store (Feast/Tecton) to compute features identically for both training and serving, write feature transforms once and reuse, use point-in-time correctness for time features, and add integration tests comparing feature values between systems.

**Q2: Explain data drift vs concept drift. How do you handle each?**

Data drift: $P(X)$ changes — input distribution shifts (new user demographics, market changes). Detect with PSI, KS test on feature distributions. Handle by retraining on recent data. Concept drift: $P(Y|X)$ changes — the relationship between inputs and outputs changes (user preferences evolve, fraud patterns change). Harder to detect without labels. Handle by monitoring prediction distribution, business metrics, and implementing regular retraining cadence. Both: implement sliding window training (prioritize recent data), online learning, or trigger-based retraining when drift exceeds thresholds.

**Q3: How do you decide between using an LLM API vs self-hosting an open-source model?**

Use API when: (1) < $500K/year in API costs, (2) need cutting-edge capabilities (GPT-4 level), (3) small team without ML infra expertise, (4) low/variable traffic that makes fixed GPU costs inefficient, (5) fast iteration is more important than cost. Self-host when: (1) data privacy requirements (healthcare, finance), (2) > $500K/year API costs, (3) need fine-grained control (custom tokenizer, architecture changes), (4) consistent high traffic justifying fixed GPU costs, (5) latency requirements that API round-trips can't meet. Hybrid is common: self-host for production traffic, API for prototyping/eval.

**Q4: What metrics would you monitor for a production RAG application?**

System: latency (TTFB, total), throughput, error rate, cost per query. Retrieval: average retrieval relevance score, number of chunks retrieved, empty retrieval rate, embedding model latency. Generation: hallucination rate (faithfulness score sampled), refusal rate, response length distribution, user satisfaction (thumbs up/down). Data: index freshness (time since last update), document count, embedding coverage. Business: task completion rate, escalation to human rate, user retention. Alert on: retrieval relevance trending down (embedding model degradation or index issues), hallucination rate spikes, latency exceeding SLA, cost per query increasing.

**Q5: Describe a canary deployment strategy for an ML model. What guardrails would you put in place?**

Canary deployment gradually shifts traffic to the new model: (1) Deploy new model alongside old model, 0% traffic. (2) Shadow mode: run both models on all requests, compare outputs without serving new model's results. Verify latency and error rates. (3) 5% canary: serve new model for 5% of users (hash-based consistent routing). Monitor for 4-24 hours. (4) Progressive rollout: 25% → 50% → 100% over days. Guardrails: automated rollback if error rate > 2× baseline, latency P99 > 1.5× baseline, or business metric drops > 2%. Require: all guardrail metrics pass for minimum duration before advancing. Additional: separate canary per region/segment to catch population-specific issues.

### System Design Questions

**Q6: Design the ML infrastructure for a startup that needs to serve 5 different ML models (recommendation, search ranking, fraud detection, content moderation, personalization) with 99.9% uptime.**

```
┌──────────────────────────────────────────────────────────────────┐
│         Multi-Model ML Platform Architecture                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  SHARED INFRASTRUCTURE                                            │
│  ┌──────────────────────────────────────────┐                    │
│  │  API Gateway (Kong / AWS API Gateway)    │                    │
│  │  → Auth, rate limiting, routing          │                    │
│  │  → /v1/recommend, /v1/rank, /v1/fraud    │                    │
│  └─────────────────┬────────────────────────┘                    │
│                    │                                              │
│  ┌─────────────────▼────────────────────────┐                    │
│  │  Model Router (service mesh / Envoy)     │                    │
│  │  → Route to correct model service        │                    │
│  │  → A/B traffic splitting                 │                    │
│  │  → Circuit breaker per model             │                    │
│  └─────────────────┬────────────────────────┘                    │
│                    │                                              │
│  MODEL SERVICES (each independently deployable)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐  │
│  │  Reco    │ │  Search  │ │  Fraud   │ │Content │ │Personl │  │
│  │  (GPU)   │ │  (CPU)   │ │  (CPU)   │ │ (GPU)  │ │ (CPU)  │  │
│  │  2 pods  │ │  4 pods  │ │  6 pods  │ │ 2 pods │ │ 3 pods │  │
│  └──────────┘ └──────────┘ └──────────┘ └────────┘ └────────┘  │
│                                                                   │
│  SHARED SERVICES                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐         │
│  │Feature Store │  │Model Registry│  │ Experiment     │          │
│  │(Feast+Redis) │  │  (MLflow)    │  │ Tracker (W&B)  │          │
│  └──────────────┘  └──────────────┘  └────────────────┘         │
│                                                                   │
│  RELIABILITY                                                      │
│  - Multi-AZ deployment (3 availability zones)                     │
│  - Per-model fallback (cached predictions or rule-based default) │
│  - Auto-scaling per model based on request queue depth            │
│  - Model-level circuit breaker (don't cascade failures)           │
│  - Health check: model-specific validation (test prediction)      │
│                                                                   │
│  CI/CD                                                            │
│  - Each model has independent deployment pipeline                 │
│  - Shared eval framework, model-specific test suites             │
│  - Canary per model (5% → 25% → 100%)                           │
│  - Automated rollback on any guardrail violation                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 15.13 Key Papers & Resources

| Resource                                                             | Year  | Why It Matters                             |
| -------------------------------------------------------------------- | ----- | ------------------------------------------ |
| _Hidden Technical Debt in ML Systems_ (Sculley et al., Google)       | 2015  | The foundational paper on ML in production |
| _ML Test Score: A Rubric for ML Production Readiness_ (Breck et al.) | 2017  | Checklist for production ML readiness      |
| _Continuous Delivery for ML_ (CD4ML, ThoughtWorks)                   | 2019  | CI/CD patterns adapted for ML              |
| _Monitoring ML Models in Production_ (Google)                        | 2020  | Practical monitoring strategies            |
| _Made with ML_ (Goku Mohandas)                                       | 2021+ | Free comprehensive MLOps course            |
| _Designing ML Systems_ (Chip Huyen)                                  | 2022  | Best book on production ML systems         |
| _The MLOps Maturity Model_ (Microsoft)                               | 2020  | Framework for assessing MLOps readiness    |
| _Feast Feature Store_                                                | 2020+ | Open-source feature store documentation    |
| _Evidently AI_                                                       | 2021+ | Open-source ML monitoring platform         |

---

[← Module 14: Generative AI](../module-14-generative-ai/README.md) | [Module 16: AI Systems Design →](../module-16-ai-systems-design/README.md)
