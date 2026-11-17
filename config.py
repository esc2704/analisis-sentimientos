from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

REAL_TRAIN = PROCESSED_DIR / "real_train.csv"
REAL_TEST = PROCESSED_DIR / "real_test.csv"

# El compañero puede dejar aquí el CSV sintético.
SYNTHETIC_PATHS = [
    PROCESSED_DIR / "synthetic_tweets.csv",
    RAW_DIR / "synthetic_tweets.csv",
]

RANDOM_SEED = 42

# Word2Vec
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
SG = 1