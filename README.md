# Real-Time Financial Fraud Detection System

This repository contains the code for the research paper

> **A Machine Learning-Based Real-Time Financial Fraud Detection Framework for Digital Transaction Monitoring**
> Inamullah Arif, Ulster University (2026). *Manuscript under review.*

It has two parts:

1. **Research experiments** (`experiments/`, `results/`): the reproducible evaluation reported in the paper.
2. **Demo application** (everything else): a FastAPI service and React frontend ([ResearchProject_Frontend](https://github.com/InamullahArif/ResearchProject_Frontend)) showing the framework working on uploaded and single transactions.

---

## Research summary

**Question.** Do behavioural features, anomaly detection and a hybrid risk score improve real-time fraud detection, and how should these signals be combined?

**Data.** Two public datasets:
- the European credit card fraud dataset (283,726 transactions, 0.17% fraud);
- the PaySim mobile-money dataset (2,770,409 transfer and cash-out transactions, 0.30% fraud).

PaySim's balance columns are excluded because its documentation says they leak the fraud label.

**Protocol.**
- Stratified 70/15/15 splits repeated over five seeds, plus a time-ordered split (train on the past, test on the future).
- Thresholds, hybrid weights and hyperparameters are chosen on validation data only.
- Models compared: logistic regression, decision tree, Random Forest, gradient boosting, Isolation Forest, an autoencoder, and the proposed hybrid score.

**Main findings.**

| Finding | Evidence |
|---|---|
| Gradient boosting gave the best PR-AUC under random splits, at about 3 ms per transaction | Credit card 0.809 ± 0.044; PaySim 0.645 ± 0.007 |
| Behavioural features for receiving accounts substantially improve detection when used as model inputs | PaySim PR-AUC 0.585 → 0.676 (random splits), 0.234 → 0.441 (time-ordered) |
| Anomaly scores did not improve detection in any form tested | Hybrid weights chose the supervised score alone in every random split. As model inputs, anomaly scores lowered PaySim PR-AUC in 5 of 5 splits |
| Graded accept / monitor / alert decisions catch fraud a single threshold misses | 61% of credit card fraud missed at the alert threshold fell in the monitor band |
| Including PaySim's balance columns gives misleading results | PR-AUC 0.998 with them, a result caused by label leakage |

Full tables are in `results/summary.json`, the frontend's **Research Results** page, and the paper.

## Reproducing the paper

Full instructions are in [`experiments/README.md`](experiments/README.md). In short:

```bash
pip install scikit-learn pandas numpy scipy matplotlib
# Put creditcard.csv and the PaySim CSV in data/ (Kaggle links in experiments/README.md)
python experiments/run_experiments.py data/creditcard.csv results/creditcard
python experiments/paysim_experiments.py data/PS_20174392719_1491204439457_log.csv results/paysim 5
python experiments/paysim_nobalance.py data/PS_20174392719_1491204439457_log.csv results/paysim
python experiments/extended_experiments.py temporal data/creditcard.csv data/PS_20174392719_1491204439457_log.csv results/extended
python experiments/extended_experiments.py anomfeat data/creditcard.csv data/PS_20174392719_1491204439457_log.csv results/extended
python experiments/extended_experiments.py tuning   data/creditcard.csv data/PS_20174392719_1491204439457_log.csv results/extended
python experiments/build_summary.py results
```

The experiment scripts use only scikit-learn, NumPy, pandas and SciPy, so no GPU is needed. They don't depend on the demo application's code.

## How the demo app relates to the paper

The demo application (`train.py`, `api/`, the frontend) and the paper's experiments (`experiments/`) are separate on purpose:

- **`experiments/`** contains the controlled evaluation behind every number in the paper: fixed protocol, repeated splits, validation-only tuning.
- **The demo app** shows how the framework would run as a service: dataset upload, batch and single-transaction scoring, hybrid risk levels, explanations. Its training script (`train.py`) supports more models (including XGBoost, LightGBM and a tabular Transformer) and is configured by default for a small sample file (`CFG.TRAIN_CSV` in `config.py`). Its outputs are a demonstration, not the paper's results.
- The app serves the paper's results through `GET /research/summary`, which the frontend displays on the **Research Results** page.

To make the demo model match the paper's best configuration, train it on the full credit card data with gradient boosting and balanced class weights, and use the supervised probability alone as the risk score. The paper found that adding anomaly scores did not help.

---

## Demo application

A machine learning pipeline and **FastAPI** service for scoring financial transactions in near real time. The project trains multiple supervised classifiers, combines them with an **Isolation Forest** anomaly signal for hybrid risk scoring, and exposes HTTP endpoints for integration with monitoring or demo clients.

---

## Overview

This repository implements a fraud-detection workflow aligned with common academic and industry practice:

- **Supervised models** learn from labelled transactions (legitimate vs fraud).
- **Unsupervised anomaly detection** highlights behaviour that deviates from normal patterns.
- **Hybrid scoring** blends supervised probability with anomaly output using configurable weights.
- **Velocity and behavioural features** (rolling windows, z-scores) enrich raw transaction fields.

Use it to experiment with class imbalance (SMOTE, class weights), compare models, and benchmark inference latency.

---

## Features


| Area           | Description                                                                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Models**     | Calibrated logistic regression, decision tree, random forest, sklearn gradient boosting, XGBoost, LightGBM, SMOTE + logistic regression, **tabular Transformer** (PyTorch), Isolation Forest |
| **Training**   | Train/validation split, PR-AUC–based selection of the best **supervised** model, metrics persisted to `models/train_report.joblib`                                                           |
| **API**        | `POST /score` with optional per-model breakdown; `GET /health`, `GET /metrics`                                                                                                               |
| **Evaluation** | Precision, recall, F1, ROC-AUC, PR-AUC, FPR, confusion matrix, single-row latency benchmark                                                                                                  |


---

## Tech Stack

- **Python 3.11+** (recommended)
- **scikit-learn**, **imbalanced-learn** (SMOTE), **XGBoost**, **LightGBM**
- **FastAPI**, **Uvicorn**, **Pydantic**
- **pandas**, **NumPy**, **joblib**, **matplotlib** (plots), **PyTorch** (tabular Transformer), **SHAP** (optional explainability experiments)

---

## Repository Layout

```
├── app.py                   # ASGI entry point (uvicorn app:app); loads api/main.py
├── api/                     # FastAPI routes (main.py) and research results endpoints (research.py)
├── experiments/             # Reproducible experiments for the paper (see experiments/README.md)
├── results/                 # Result files for every experiment + summary.json
├── train.py                 # Model training and bundle export
├── evaluate.py              # Offline metrics, fairness slices, cost-threshold search
├── ablation.py              # Train with / without velocity features
├── explain.py               # SHAP (trees) + LIME (tabular) explanations
├── dataset_loaders.py       # CSV presets: generic, paysim, uci_creditcard
├── anomaly_autoencoder.py   # PyTorch reconstruction-error anomaly scores
├── generate_synthetic_stress.py  # Synthetic stress-test CSV
├── streaming/kafka_consumer_demo.py  # Optional Kafka listener stub
├── features.py              # Cleaning + velocity features + preprocessor (scaled numerics)
├── runtime_env.py           # OpenMP env (import first)
├── model_io.py              # Safe joblib loading + NumPy mismatch hints
├── tabular_transformer.py   # PyTorch tabular Transformer
├── config.py                # Paths, columns, thresholds, dataset preset, costs
├── plot_*.py                # Training curves, metrics, threshold trade-offs
├── data/                    # CSV datasets
└── models/                  # Generated artifacts (see .gitignore)
```

---

## Prerequisites

- Python **3.10+** (3.11 used in development)
- A CSV with at least: `timestamp`, `user_id`, `amount`, `is_fraud`, plus optional `V1`…`V28` for credit-card–style features

Adjust column names in `config.py` if your dataset differs.

Install **PyTorch** for the tabular Transformer (see [pytorch.org](https://pytorch.org) for a CUDA build if you use a GPU). The `requirements.txt` entry installs the CPU wheel by default; upgrade if you need GPU training.

---

## Tabular Transformer (optional)

This project includes an **FT-Transformer–style** model (`tabular_transformer.py`):

- Each numeric column after preprocessing is projected to a fixed `d_model` dimension (one “token” per feature).
- Learned positional bias and a **TransformerEncoder** stack model interactions between features.
- Outputs are calibrated with **sigmoid** for fraud probability; training uses **BCEWithLogitsLoss** with **class-balanced** positive weight and validation **average precision** for checkpoint selection.

It is **not** a text/NLP Transformer: it operates on the same tabular feature matrix as the sklearn models. Disable it (faster training, fewer dependencies) with `TRANSFORMER_ENABLED = False` in `config.py`. If PyTorch is not installed, training skips the Transformer and continues with tree/linear models only.

---

## Installation

```bash
cd fraud_detection_system-main
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

### Troubleshooting: `BitGenerator` / cannot load `*.joblib`

If you see `MT19937 is not a known BitGenerator` (or similar) when running `evaluate.py` or starting the API, the model files were saved with a **different NumPy version** than the one you use now (common after pinning `numpy<2`).

**Fix:** delete the old artifacts and retrain:

```bash
rm models/fraud_model.joblib models/scoring_bundle.joblib models/train_report.joblib
python train.py
```

(Adjust paths on Windows, or delete those files in Explorer.)

### OpenMP / `libiomp` and `libomp` warning

If you see a **threadpoolctl** warning about **Intel OpenMP** and **LLVM OpenMP** together (common when **PyTorch** and **scikit-learn** share one process), the project sets `KMP_DUPLICATE_LIB_OK=TRUE` via `runtime_env.py`, imported first in `train.py`, `evaluate.py`, `app.py`, and `tabular_transformer.py`. On **Linux** servers, if you still see deadlocks, try limiting threads (e.g. `export OMP_NUM_THREADS=1`) or use a single BLAS/OpenMP stack (see the [threadpoolctl note](https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md)).

---

## Usage

### 1. Train models

Place training data at `data/transactions.csv`. Set `**DATASET_PRESET`** in `config.py` to match the file:

- `generic` — already matches project columns (`timestamp`, `user_id`, `amount`, `is_fraud`, …)
- `paysim` — typical PaySim column names (`isFraud`, `nameOrig`, …) are mapped in `dataset_loaders.py`
- `uci_creditcard` — raw ULB file with `Time`, `Amount`, `Class`

```bash
python train.py
```

Categorical columns listed in `**CATEGORICAL_FEATURES**` (e.g. `merchant`, `category`, `location`) are one-hot encoded when present. Numeric columns use **imputation + `StandardScaler`** when `SCALE_NUMERIC = True`.

Training adds **random undersampling + LR** (`rus_lr`) when `UNDERSAMPLE_ENABLED = True`, and a **PyTorch autoencoder** anomaly scorer when `AUTOENCODER_ENABLED = True` (blended with Isolation Forest using `ANOMALY_BLEND_ISO` / `ANOMALY_BLEND_AE`).

Outputs include:

- `models/fraud_model.joblib` — best supervised model (by PR-AUC)
- `models/scoring_bundle.joblib` — best model, all supervised estimators, Isolation Forest, metrics, column lists
- `models/train_report.joblib` — evaluation cache for plotting scripts
- `models/model_metrics.csv` — **Accuracy**, PR-AUC, ROC-AUC, and F1 for **every** model on the hold-out split (including `tab_transformer` when trained)

Training also prints a summary table to the terminal.

### 2. Evaluate on a hold-out set

```bash
python evaluate.py
```

Default test path is `data/test.csv`. The report includes **Accuracy** plus PR-AUC, precision, recall, F1, and FPR for the **deployed** (best) model with optional hybrid anomaly blending.

**Accuracy for all models (including tabular Transformer):**

```bash
python evaluate.py --all-models
python evaluate.py --all-models path/to/labelled.csv
```

Writes `models/evaluate_all_models_metrics.csv` with Accuracy, PR-AUC, ROC-AUC, and F1 per model at `ALERT_THRESHOLD`.

**Fairness-style slices (amount quartiles)** — simple segment analysis (not demographic data):

```bash
python evaluate.py --fairness
```

**Cost-sensitive threshold** — grid search using `COST_FALSE_NEGATIVE` and `COST_FALSE_POSITIVE` in `config.py`:

```bash
python evaluate.py --cost-threshold
```

### 3. Ablation, synthetic data, streaming stub, explanations

```bash
python ablation.py                 # default: full velocity features
python ablation.py --no-velocity   # ablation without rolling / z-score features

python generate_synthetic_stress.py   # writes data/stress_synthetic.csv

# Optional: after installing kafka-python and running a broker
python streaming/kafka_consumer_demo.py

# LIME (default) or SHAP for tree best-model (needs sample CSV with same schema as training)
FRAUD_EXPLAIN_MODE=lime python explain.py
FRAUD_EXPLAIN_MODE=shap python explain.py
```

### 4. API authentication (optional)

If the environment variable `**FRAUD_API_KEY**` is set, `POST /score` and `GET /metrics` require header `**X-API-Key**`. `GET /health` stays unauthenticated and reports whether a key is configured.

### 5. Generate plots

After training:

```bash
python plot_training_results.py
python plot_curves.py
python plot_advanced_metrics.py
```

Figures and CSVs are written under `models/`.

### 6. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- **Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Score a transaction:** `POST /score` with JSON body matching the `Transaction` schema in `api/main.py`
- **Paper results:** `GET /research/summary` (all datasets) or `GET /research/summary/{creditcard|paysim}`
- **Per-model probabilities:** `POST /score?breakdown=true` (slower; loads all supervised pipelines from the bundle)

---

## Configuration (`config.py`)


| Setting                                                          | Role                                                                               |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `TARGET_COL`, `TIME_COL`, `USER_COL`, `AMOUNT_COL`               | Column names for labels and feature engineering                                    |
| `ALERT_THRESHOLD`                                                | Score above which an alert is raised                                               |
| `MEDIUM_RISK_THRESHOLD`, `HIGH_RISK_THRESHOLD`                   | Bands for `risk_level`                                                             |
| `HYBRID_WEIGHT_ML`, `HYBRID_WEIGHT_ANOMALY`                      | Blend of supervised vs anomaly score (should sum to ~1.0)                          |
| `TRANSFORMER_ENABLED`                                            | If `True`, train and evaluate the PyTorch tabular Transformer (`tab_transformer`)  |
| `TRANSFORMER_D_MODEL`, `TRANSFORMER_NHEAD`, `TRANSFORMER_LAYERS` | Transformer width, attention heads, depth (`d_model` must be divisible by `nhead`) |
| `TRANSFORMER_EPOCHS`, `TRANSFORMER_BATCH_SIZE`, `TRANSFORMER_LR` | Training schedule for the tabular Transformer                                      |
| `TRANSFORMER_VAL_FRACTION`                                       | Hold-out fraction for validation metrics during Transformer training               |


---

## Evaluation Metrics

The training and evaluation scripts report **precision**, **recall**, **F1**, **ROC-AUC**, **PR-AUC** (important under class imbalance), and **false positive rate**. For fraud detection, **recall** and **PR-AUC** are often more informative than accuracy alone.

---

## Future Enhancements

1. **Streaming ingestion** — Integrate **Apache Kafka** or **Redis** for shared transaction history instead of in-memory `HISTORY` in `app.py` (required for multi-worker production).
2. **Explainability** — Use **SHAP** or **LIME** on the deployed tree model with a small background sample; add `POST /explain` with latency limits or async jobs.
3. **Concept drift** — Periodic retraining, sliding windows, and monitoring of score distributions and label delay.
4. **Threshold tuning** — Cost-sensitive thresholds using business costs (false negative vs false positive) rather than a single global `ALERT_THRESHOLD`.
5. **Additional anomaly detectors** — One-Class SVM, LOF, or shallow autoencoders; compare PR-AUC and latency vs Isolation Forest.
6. **Deploy bundle** — Ship a slim `scoring_bundle` with only the best model + anomaly detector to reduce load time and memory.
7. **Graph or sequence models** — For account–merchant–device graphs or strict sequence data, explore specialised architectures (separate from this tabular baseline).
8. **Sequence / NLP Transformers** — If raw transaction descriptions or merchant text become available, fine-tune a small language model or build a multimodal stack (tabular Transformer + text encoder).
9. **Deeper tabular Transformers** — Experiment with **FT-Transformer**, **SAINT**, or **TabPFN**-style priors; tune regularisation for small fraud samples.

---

## Improvements & Suggestions

- **Scaling** — The tabular Transformer path applies **StandardScaler** after the column preprocessor; tree pipelines can still benefit from consistent scaling for linear baselines.
- **Transformer training** — Increase `TRANSFORMER_EPOCHS` cautiously on large CSVs (runtime grows); for thesis runs, pin `torch` and log validation PR-AUC per epoch if you extend the training script.
- **Validation** — Use stratified **k-fold** or **time-series split** if timestamps reflect real ordering; avoid random splits that leak future behaviour.
- **Imbalance** — Compare **SMOTE** only in an inner CV loop to reduce overfitting; try **class_weight** alone first on strong tree models.
- **API hardening** — Rate limiting, authentication, request size limits, and structured logging for audit trails.
- **Tests** — Add `pytest` smoke tests for `features.py` and a minimal `POST /score` response shape.
- **Reproducibility** — Pin dependency versions in `requirements.txt` (e.g. `scikit-learn==x.y.z`) for thesis submission or deployment snapshots.

---

## Acknowledgements

Public datasets commonly used in fraud-detection research include the **Kaggle Credit Card Fraud** dataset, **IEEE-CIS**, **PaySim**, and **UCI** financial benchmarks. This project is intended for **research and education**; production systems require legal review, privacy safeguards, and operational monitoring.

---

## Author

**Inamullah Arif**, MSc Computer Science, Ulster University (Manchester campus).

For questions about the methodology or about reproducing the results, please open an issue in this repository.

---

## License

Released under the [MIT License](LICENSE). The datasets are not included and remain subject to their own licences (see `experiments/README.md`).