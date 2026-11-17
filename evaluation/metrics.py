import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR


def evaluate_model(y_true, y_pred, model_name, scenario):

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model": model_name,
        "scenario": scenario,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
    }

    filename = f"{model_name}_{scenario}"

    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / f"{filename}.csv",
        index=False
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Negativo", "Positivo"]
    )

    display.plot(cmap="Blues")
    plt.title(f"{model_name} - {scenario}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{filename}_confusion.png")
    plt.close()

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "real": y_true,
        "predicted": y_pred
    }).to_csv(
        PREDICTIONS_DIR / f"{filename}_predictions.csv",
        index=False
    )

    print(f"\nResultados: {model_name} - {scenario}")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")

    return metrics