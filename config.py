from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

# Dataset Sentiment140

SEN_URL = "http://help.sentiment140.com/home"
SEN_DATA_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

SEN_TRAIN_FILE_NAME = "training.1600000.processed.noemoticon.csv"

# Dataset Synthetic Tweets Qwen3-30B-A3B-Instruct-2507-FP8

SYN_URL = "https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
SYN_DATA_URL = "https://drive.canga.cc/public.php/dav/files/cdn-synthetict-tweets/?accept=zip"

SYN_TRAIN_FILE_NAME = "synthetic_tweets_en_1600000.csv"


DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

RESULTS_DIR = ROOT_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

REAL_TRAIN = PROCESSED_DIR / "real_train.csv"
REAL_TEST = PROCESSED_DIR / "real_test.csv"
REAL_VAL = PROCESSED_DIR / "real_val.csv"

SYN_TRAIN = PROCESSED_DIR / "synthetic_train.csv"
SYN_TEST = PROCESSED_DIR / "synthetic_test.csv"
SYN_VAL = PROCESSED_DIR / "synthetic_val.csv"

# Proporciones de los splits (sobre el total): 70% train, 15% val, 15% test
VAL_SIZE = 0.15
TEST_SIZE = 0.15


RANDOM_SEED = 42

# Word2Vec
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
SG = 1