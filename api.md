# Fraud Detection API

Base URL (local): `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs` (Swagger UI)

## Authentication

If the environment variable `FRAUD_API_KEY` is set, protected routes require:

| Header        | Value              |
|---------------|--------------------|
| `X-API-Key`   | Same as `FRAUD_API_KEY` |

If `FRAUD_API_KEY` is **not** set, those routes do not require a key.

**Public (no key):** `GET /health`, `POST /upload-dataset`  
**Protected (key when configured):** `POST /predict`, `GET /analytics`, `POST /predict-single`, and the routes listed in [Additional endpoints](#additional-endpoints).

---

## GET /health

Public (no API key). Liveness and **which model artifact** is driving supervised scores.

### Response `200`

```json
{
  "status": "ok",
  "best_model": "fraud_model_credit_card",
  "bundle_loaded": false,
  "model_artifact": "/path/to/project/models/fraud_model_credit_card.pkl",
  "api_key_required": false,
  "supervised_models_available": []
}
```

| Field | Description |
|--------|-------------|
| `best_model` | Name of the active supervised head: bundle’s `best_model_name`, or the **stem** of the standalone file (e.g. `fraud_model_credit_card` for `fraud_model_credit_card.pkl`). |
| `bundle_loaded` | `true` if `models/scoring_bundle.joblib` was loaded (and `USE_SCORING_BUNDLE` is enabled in `config.py`). |
| `model_artifact` | Absolute path of the loaded file: `scoring_bundle.joblib` when using the bundle, otherwise the standalone pipeline (default `models/fraud_model_credit_card.pkl`, configurable — see [Model files](#model-files)). |
| `supervised_models_available` | Keys from the bundle’s `supervised_models` map; empty when only a standalone pipeline is loaded. |

---

## 1. `POST /upload-dataset`

Upload a CSV of transactions. The server runs the same preprocessing as training (`engineer_features`) and stores the result in memory for the next `/predict` and `/analytics` calls.

### Request

- **Content-Type:** `multipart/form-data`
- **Fields:**

| Field        | Type   | Required | Description |
|-------------|--------|----------|-------------|
| `file`      | file   | yes      | CSV file (UTF-8). |
| `preset`    | string | no       | Default `generic`. Use `paysim` or `uci_creditcard` to remap columns to the project schema (see `dataset_loaders.py`). |

### Expected CSV columns

Minimum: `timestamp`, `user_id`, `amount`, plus any optional categorical columns (`merchant`, `channel`, etc.).  
If present, `is_fraud` (or `CFG.TARGET_COL`) is used only for analytics; it is dropped during prediction.

### Response `200`

```json
{
  "upload_id": "<uuid>",
  "rows": 1000,
  "columns": ["timestamp", "user_id", "amount", ...],
  "has_target": true,
  "preset": "generic"
}
```

### Errors

| Code | When |
|------|------|
| 400  | Empty file |
| 400  | Non-UTF-8 CSV |
| 422  | Preprocessing / feature engineering failed |

---

## 2. `POST /predict`

Runs batch inference on the **last uploaded** dataset (from `POST /upload-dataset`). Computes hybrid scores (supervised model + anomaly blend) and caches results for `GET /analytics`.

### Request

- **Content-Type:** none required (no body)
- **Query parameters:**

| Parameter   | Type   | Required | Description |
|------------|--------|----------|-------------|
| `threshold` | float | no       | Binary fraud threshold. Default: `CFG.ALERT_THRESHOLD` (e.g. `0.85`). |

### Headers (when `FRAUD_API_KEY` is set)

`X-API-Key: <your-key>`

### Response `200`

```json
{
  "total_transactions": 1000,
  "fraud_count": 12,
  "fraud_percentage": 1.2,
  "average_risk_score": 0.084,
  "threshold": 0.85,
  "avg_latency_ms_per_row": 0.15,
  "model_artifact": "/path/to/project/models/fraud_model_credit_card.pkl",
  "best_model": "fraud_model_credit_card"
}
```

- **fraud_count / fraud_percentage:** Rows with hybrid risk score ≥ `threshold`.
- **average_risk_score:** Mean hybrid score in `[0, 1]`.
- **model_artifact:** Same meaning as `GET /health` — filesystem path of the bundle or standalone model used for `predict_proba`.
- **best_model:** Same meaning as `GET /health` (`best_model` field).

### Errors

| Code | When |
|------|------|
| 400  | No dataset uploaded yet (`POST /upload-dataset` first) |

---

## 3. `GET /analytics`

Returns fraud distribution, risk histogram, and top suspicious transactions for the **last** `POST /predict` run on the current upload.

### Request

- **Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k`   | int  | 20      | Number of highest-risk rows to return (1–200). |
| `bins`    | int  | 20      | Histogram bins for risk scores (5–100). |

### Headers (when `FRAUD_API_KEY` is set)

`X-API-Key: <your-key>`

### Response `200`

```json
{
  "fraud_distribution": {
    "ground_truth_fraud_rate": 0.01,
    "predicted_fraud_rate": 0.012,
    "labelled_rows": 1000
  },
  "risk_score_histogram": {
    "bins": 20,
    "counts": [10, 20, ...],
    "edges": [0.0, 0.05, ..., 1.0]
  },
  "top_suspicious_transactions": [
    {
      "risk_score": 0.97,
      "row_index": 42,
      "timestamp": "...",
      "user_id": "...",
      "amount": 5000.0,
      "merchant": "...",
      "channel": "..."
    }
  ],
  "model_feature_importance": {
    "amount": 0.12,
    "rolling_tx_count_1h": 0.08
  }
}
```

- If the upload **has no** label column (`is_fraud`), `fraud_distribution` uses `predicted_fraud_count`, `predicted_fraud_rate`, and `note: "No ground-truth column in upload"`.
- `model_feature_importance` is present only when the training bundle includes `tree_pipeline_for_shap` (Random Forest pipeline).

### Errors

| Code | When |
|------|------|
| 400  | Upload exists but `POST /predict` was not run yet |

---

## 4. `POST /predict-single`

Scores **one** transaction as JSON. Appends to an in-memory history (for velocity and rolling features), so order and user context matter. Returns hybrid fraud probability and a binary label.

### Request

- **Content-Type:** `application/json`
- **Body:** JSON object (see below).

### Required fields

| Field       | Type   | Description |
|------------|--------|-------------|
| `timestamp` | string | ISO datetime string (e.g. `2024-01-15T10:30:00`). |
| `user_id`   | string | User identifier. |
| `amount`    | number | Transaction amount. |

### Optional fields

`merchant`, `category`, `location`, `device_type`, `channel` (all nullable strings).  
Extra fields are allowed (e.g. `V1`…`V28` for credit-card style features); missing `V*` default to `0`.

### Query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `explain` | bool | false   | If `true`, includes SHAP-style feature contributions and RF importances (requires `tree_pipeline_for_shap` in bundle). |

### Headers (when `FRAUD_API_KEY` is set)

`X-API-Key: <your-key>`

### Response `200`

```json
{
  "fraud_probability": 0.73,
  "label": "fraud",
  "ml_probability": 0.71,
  "anomaly_score": 0.82,
  "risk_level": "high",
  "model_artifact": "/path/to/project/models/fraud_model_credit_card.pkl",
  "feature_importance": {},
  "shap_explanations": [],
  "latency_ms": 12.345
}
```

- **fraud_probability:** Hybrid score in `[0, 1]` (ML + anomaly weights from `config`).
- **label:** `"fraud"` if `fraud_probability >= ALERT_THRESHOLD`, else `"legitimate"`.
- **risk_level:** `"low"` | `"medium"` | `"high"` (bands from `config`).
- **model_artifact:** Same meaning as `GET /health` (path of `scoring_bundle.joblib` or standalone `.pkl` / `.joblib` used for **ml_probability**). Omitted or `null` only if no artifact was resolved (abnormal after a successful startup).
- **shap_explanations / feature_importance:** Populated when `explain=true` and the tree SHAP pipeline is available.

### Example (`curl`)

```bash
curl -s -X POST "http://127.0.0.1:8000/predict-single?explain=false" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key-if-set>" \
  -d '{"timestamp":"2024-06-01T12:00:00","user_id":"U1","amount":120.5,"merchant":"M1","channel":"online"}'
```

---

## Typical flow

1. `POST /upload-dataset` with a CSV file.  
2. `POST /predict` (optional `threshold`).  
3. `GET /analytics` for charts and top risky rows.  

For real-time scoring without a batch upload, use `POST /predict-single` (and/or legacy `POST /score`).

---

## Additional endpoints

| Method | Path | Auth | Notes |
|--------|------|------|--------|
| GET | `/health` | no | Liveness; includes `model_artifact` and `best_model` (see [GET /health](#get-health)). |
| GET | `/metrics` | yes | Training-time metrics from `scoring_bundle.joblib`. |
| POST | `/score` | yes | Legacy real-time score + reasons + optional per-model breakdown. |
| POST | `/reload-models` | yes | Reload scoring artifacts from disk (see below). |
| GET | `/queue-stats` | yes | Async scoring runner statistics. |

### `POST /reload-models`

Reloads the same files the process used at startup (after you replace `scoring_bundle.joblib` and/or the standalone model on disk). No body.

### Response `200`

```json
{
  "status": "reloaded",
  "best_model": "fraud_model_credit_card",
  "model_artifact": "/path/to/project/models/fraud_model_credit_card.pkl"
}
```

| Field | Description |
|--------|-------------|
| `status` | Always `"reloaded"` on success. |
| `best_model` | Same semantics as `GET /health`. |
| `model_artifact` | Path of the bundle or standalone file now in use. |

---

## Model files

Inference prefers **`models/scoring_bundle.joblib`** when it exists and `USE_SCORING_BUNDLE` is `true` in `config.py`. Otherwise it loads a **standalone** supervised pipeline:

- Default file: **`models/fraud_model_credit_card.pkl`** (`CFG.FRAUD_MODEL_FILE`).
- If that file is missing, the server may fall back to **`models/fraud_model.joblib`** (legacy).

Override the standalone path with environment variable **`FRAUD_MODEL_PATH`** (absolute path, or relative to `models/`). Set **`USE_SCORING_BUNDLE = false`** in `config.py` to force the standalone file even when a bundle exists.

---

## Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**CORS:** By default the API allows browser requests from **`http://localhost:3000`** and **`http://127.0.0.1:3000`**. Override with **`CORS_ALLOW_ORIGINS`**: comma-separated list of origins, or `*` for any origin (no credentials).

Ensure model artifacts exist: run `python train.py` for `fraud_model.joblib` / `scoring_bundle.joblib`, or place a standalone pipeline at `models/fraud_model_credit_card.pkl` (or configure `FRAUD_MODEL_FILE` / `FRAUD_MODEL_PATH`). See [Model files](#model-files).
