# Research experiments

This folder holds the scripts that produce every result in the paper
*"A Machine Learning-Based Real-Time Financial Fraud Detection Framework for Digital Transaction Monitoring"*.
The results are saved in `results/` and shown in the frontend's **Research Results** page.

## Datasets

Neither dataset is stored in this repository. Download them and place the CSVs in `data/`:

| Dataset | Download | File to put in `data/` |
|---|---|---|
| Credit Card Fraud Detection (ULB) | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `creditcard.csv` |
| PaySim mobile money | https://www.kaggle.com/datasets/ealaxi/paysim1 | `PS_20174392719_1491204439457_log.csv` |

PaySim note: the four balance columns are excluded from all model inputs because the dataset's documentation says they leak the fraud label.

## Setup

```
pip install scikit-learn pandas numpy scipy matplotlib
```

Python 3.10 or newer. All models come from scikit-learn, so no GPU is needed.

## Run everything

Run these from the backend project folder (`fraud_detection_system-main`):

```
# 1. Main comparison: 7 models, ablation, imbalance strategies (5 stratified 70/15/15 splits)
python experiments/run_experiments.py data/creditcard.csv results/creditcard
python experiments/paysim_experiments.py data/PS_20174392719_1491204439457_log.csv results/paysim 5

# 2. PaySim behavioural-feature check with cost-sensitive weights
python experiments/paysim_nobalance.py data/PS_20174392719_1491204439457_log.csv results/paysim

# 3. Robustness checks
python experiments/extended_experiments.py temporal data/creditcard.csv data/PS_20174392719_1491204439457_log.csv results/extended
python experiments/extended_experiments.py anomfeat data/creditcard.csv data/PS_20174392719_1491204439457_log.csv results/extended
python experiments/extended_experiments.py tuning   data/creditcard.csv data/PS_20174392719_1491204439457_log.csv results/extended

# 4. Combine everything into results/summary.json (used by the API and frontend)
python experiments/build_summary.py results
```

On a typical laptop, step 1 takes about 1 hour, step 2 about 20 minutes, and step 3 about 1.5 hours. Every script saves its results after each split, so a stopped run keeps its partial results.

## What each script does

| Script | Purpose |
|---|---|
| `run_experiments.py` | Credit card experiments: logistic regression, decision tree, Random Forest, gradient boosting, Isolation Forest, autoencoder, and the hybrid risk score; ablation; imbalance strategies; latency and throughput; permutation importance |
| `paysim_experiments.py` | The same on PaySim, plus causal receiver-account behavioural features (earlier transactions, 24-hour velocity, amount deviation, new-account flag) and a label-leakage check |
| `paysim_nobalance.py` | Effect of the behavioural features when models use cost-sensitive weights |
| `extended_experiments.py` | `temporal`: train on the past, test on the future. `anomfeat`: anomaly scores as model inputs. `tuning`: validation-based hyperparameter search for gradient boosting |
| `build_summary.py` | Combines all result files into `results/summary.json` |
| `view_results.py` | Prints tables and saves CSV and PNG charts for one results file, e.g. `python experiments/view_results.py results/paysim/results.json` |

## Viewing the results in the app

1. Start the backend: `uvicorn api.main:app --reload`
2. Open `http://127.0.0.1:8000/research/summary` to see the raw JSON, or `/docs` for the API page.
3. Start the frontend (`npm run dev` in the frontend project) and open **Research Results** in the navigation bar.

## Reproducibility

- Splits are stratified and seeded (seeds 0–4). The time-ordered split is deterministic, and model randomness is repeated over seeds 0–2.
- Hyperparameters, hybrid weights and decision thresholds are chosen on the validation partition only. Each test partition is used once.
- Small differences in the third decimal place across machines and library versions are expected. Latency and throughput depend on hardware.
