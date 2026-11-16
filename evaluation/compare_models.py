import pandas as pd

from config import METRICS_DIR


def compare_models():

    files = list(METRICS_DIR.glob("*.csv"))

    if not files:
        print("Todavía no hay resultados.")
        return

    results = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    results = results.sort_values(
        by="f1_macro",
        ascending=False
    )

    print("\nCOMPARACIÓN DE MODELOS\n")
    print(results.to_string(index=False))

    results.to_csv(
        METRICS_DIR / "comparison.csv",
        index=False
    )


if __name__ == "__main__":
    compare_models()