import pandas as pd

from config import (
    REAL_TRAIN,
    REAL_TEST,
    SYNTHETIC_PATHS,
    VECTOR_SIZE,
    WINDOW,
    MIN_COUNT,
    WORKERS,
    SG,
    RANDOM_SEED
)

from models.baseline import create_baseline
from models.word2vec import Word2VecClassifier
from evaluation.metrics import evaluate_model


def load_dataset(path):

    df = pd.read_csv(path)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError(
            f"{path} debe contener las columnas 'text' y 'label'."
        )

    df = df.dropna(subset=["text", "label"]).copy()

    df["text"] = df["text"].astype(str)

    # Sentiment140: 0 = negativo, 4 = positivo.
    df["label"] = df["label"].replace({4: 1})

    df["label"] = pd.to_numeric(df["label"])

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(
            f"Etiquetas no válidas en {path}: {df['label'].unique()}"
        )

    df["label"] = df["label"].astype(int)

    return df


def load_synthetic():

    for path in SYNTHETIC_PATHS:
        if path.exists():
            print(f"Dataset sintético encontrado: {path}")
            return load_dataset(path)

    raise FileNotFoundError(
        "No se encuentra synthetic_tweets.csv. "
        "Colócalo en data/processed/ o data/raw/."
    )


def run_experiment(scenario, model_name):

    real_train = load_dataset(REAL_TRAIN)
    real_test = load_dataset(REAL_TEST)

    if scenario == "real":
        train = real_train

    elif scenario == "synthetic":
        train = load_synthetic()

    elif scenario == "combined":
        synthetic = load_synthetic()

        train = pd.concat(
            [real_train, synthetic],
            ignore_index=True
        )

        train = train.sample(
            frac=1,
            random_state=RANDOM_SEED
        ).reset_index(drop=True)

    else:
        raise ValueError(f"Escenario desconocido: {scenario}")

    X_train = train["text"].tolist()
    y_train = train["label"].tolist()

    X_test = real_test["text"].tolist()
    y_test = real_test["label"].tolist()

    if model_name == "bow":
        model = create_baseline()

    elif model_name == "word2vec":
        model = Word2VecClassifier(
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            workers=WORKERS,
            sg=SG,
            random_state=RANDOM_SEED
        )

    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    print(f"\nEntrenando {model_name} con escenario {scenario}...")
    print(f"Ejemplos de entrenamiento: {len(X_train)}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    evaluate_model(
        y_test,
        predictions,
        model_name,
        scenario
    )