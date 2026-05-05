# Command cheat sheet

Run from the project folder: `cd /Users/mac/Desktop/Inam-Project-UK`

## Setup (once)

```bash
cd /Users/mac/Desktop/Inam-Project-UK
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

*(Windows activate: `env\Scripts\activate`)*

---

## Data prep (`scripts.py`)

```bash
python scripts.py clean-creditcard
python scripts.py clean-creditcard --max-benign 10000 --output data/creditcard_clean_small.csv

python scripts.py sample-transactions
python scripts.py sample-transactions --n 100 --input data/transactions.csv --output data/transactions_100.csv
python scripts.py sample-transactions --n 100 --stratify-col is_fraud --seed 42
```

---

## Training & models

```bash
python train.py
```

*(Uses `CFG.TRAIN_CSV` in `config.py` — currently `data/transactions_100.csv`.)*

**If joblib / NumPy errors after upgrades:**

```bash
rm models/fraud_model.joblib models/scoring_bundle.joblib models/train_report.joblib
python train.py
```

---

## Evaluation (`evaluate.py`)

```bash
python evaluate.py
python evaluate.py data/test.csv
python evaluate.py data/transactions_100.csv

python evaluate.py --all-models
python evaluate.py --all-models data/transactions_100.csv

python evaluate.py --fairness
python evaluate.py --fairness data/test.csv

python evaluate.py --cost-threshold
python evaluate.py --cost-threshold data/test.csv
```

---

## Ablation, synthetic data, streaming stub

```bash
python ablation.py
python ablation.py --no-velocity

python generate_synthetic_stress.py

python streaming/kafka_consumer_demo.py
```

---

## Explainability (`explain.py`)

```bash
FRAUD_EXPLAIN_MODE=lime python explain.py
FRAUD_EXPLAIN_MODE=shap python explain.py

FRAUD_EXPLAIN_CSV=data/transactions_100.csv FRAUD_EXPLAIN_ROW=0 FRAUD_EXPLAIN_MODE=lime python explain.py
```

---

## Plots (after training)

```bash
python plot_training_results.py
python plot_curves.py
python plot_advanced_metrics.py
```

---

## API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

- Docs: `http://127.0.0.1:8000/docs`
- Optional auth: set `FRAUD_API_KEY`, then send header `X-API-Key` on protected routes (see README).

---

## Credit-card → `transactions.csv` (if you use that script)

```bash
python prepare_creditcard.py
```

---

## Config reminder

CSV paths and preset are in `config.py` — `TRAIN_CSV` for `train.py` / `ablation.py` / LIME background in `explain.py`; `DATASET_PRESET` (`generic` | `paysim` | `uci_creditcard`) must match your CSV layout.
