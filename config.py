from dataclasses import dataclass
from pathlib import Path
@dataclass(frozen=True)
class Config:
    # Paths
    ROOT: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = ROOT / "data"
    MODELS_DIR: Path = ROOT / "models"
    # When `scoring_bundle.joblib` exists, it is loaded first unless USE_SCORING_BUNDLE is False.
    USE_SCORING_BUNDLE: bool = True
    # Standalone supervised artifact under MODELS_DIR if no bundle (or bundle disabled).
    # Override path with env FRAUD_MODEL_PATH (absolute, or relative to MODELS_DIR).
    FRAUD_MODEL_FILE: str = "fraud_model_credit_card.pkl"
    # Default CSV for `train.py` / `evaluate.py` / `explain.py` / `ablation.py`
    TRAIN_CSV: Path = DATA_DIR / "transactions_100.csv"

    # Data
    TARGET_COL: str = "is_fraud"     # rename to match your dataset
    TIME_COL: str = "timestamp"      # rename to match your dataset
    USER_COL: str = "user_id"        # rename to match your dataset
    AMOUNT_COL: str = "amount"       # rename to match your dataset

    # Train
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    # Fewer trees / epochs for quicker iteration and faster API bundle refresh (set False for full runs)
    FAST_TRAIN: bool = True

    # Smaller models/scoring_bundle.joblib: only persist best supervised model + omit SHAP RF clone.
    # Set False if you need every model in the API breakdown or tree_pipeline_for_shap in the bundle.
    SLIM_SCORING_BUNDLE: bool = True
    # zlib level 0–9 for joblib dumps (0 = no compression; 3 is a good size/speed tradeoff)
    JOBLIB_COMPRESS: int = 3

    # Dataset CSV preset for `dataset_loaders.load_dataset` (see dataset_loaders.py)
    DATASET_PRESET: str = "generic"  # generic | paysim | uci_creditcard

    # If present in the dataframe, these columns are one-hot encoded (merchant / channel / PaySim type)
    CATEGORICAL_FEATURES: tuple[str, ...] = (
        "merchant",
        "category",
        "location",
        "device_type",
        "channel",
        "tx_type",
        "counterparty_id",
    )

    SCALE_NUMERIC: bool = True
    UNDERSAMPLE_ENABLED: bool = True
    AUTOENCODER_ENABLED: bool = True
    AUTOENCODER_EPOCHS: int = 25
    FAST_AUTOENCODER_EPOCHS: int = 8

    # Ensemble over LR (calibrated) + RF + XGB: "voting" | "stacking" | "none"
    ENSEMBLE_MODE: str = "voting"

    # When both IsoForest and autoencoder are available, blend their [0,1] anomaly scores:
    ANOMALY_BLEND_ISO: float = 0.5
    ANOMALY_BLEND_AE: float = 0.5

    # Real-time threshold
    ALERT_THRESHOLD: float = 0.85
    MEDIUM_RISK_THRESHOLD: float = 0.30
    HIGH_RISK_THRESHOLD: float = 0.60

    # Hybrid scoring weights (must sum close to 1)
    HYBRID_WEIGHT_ML: float = 0.80
    HYBRID_WEIGHT_ANOMALY: float = 0.20

    # Tabular Transformer (PyTorch — feature tokens + Transformer encoder)
    TRANSFORMER_ENABLED: bool = True
    TRANSFORMER_D_MODEL: int = 64
    TRANSFORMER_NHEAD: int = 4
    TRANSFORMER_LAYERS: int = 2
    TRANSFORMER_DROPOUT: float = 0.1
    TRANSFORMER_EPOCHS: int = 30
    # When FAST_TRAIN: tabular Transformer uses this epoch cap instead of TRANSFORMER_EPOCHS
    FAST_TRANSFORMER_EPOCHS: int = 10
    TRANSFORMER_BATCH_SIZE: int = 2048
    TRANSFORMER_LR: float = 1e-3
    TRANSFORMER_WEIGHT_DECAY: float = 1e-4
    TRANSFORMER_VAL_FRACTION: float = 0.1
    # Focal loss for tabular Transformer (alternative to pos-weighted BCE)
    TRANSFORMER_FOCAL_LOSS: bool = False
    TRANSFORMER_FOCAL_GAMMA: float = 2.0
    TRANSFORMER_FOCAL_ALPHA: float = 0.25

    # Optional cost weights for threshold search (evaluate.py --cost-threshold)
    COST_FALSE_NEGATIVE: float = 100.0
    COST_FALSE_POSITIVE: float = 1.0

CFG = Config()
CFG.MODELS_DIR.mkdir(parents=True, exist_ok=True)