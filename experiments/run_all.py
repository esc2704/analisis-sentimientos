from experiments.common import run_experiment
from evaluation.compare_models import compare_models


def main():

    scenarios = [
        "real",
        "synthetic",
        "combined"
    ]

    models = [
        "bow",
        "word2vec"
    ]

    for scenario in scenarios:
        for model in models:
            run_experiment(scenario, model)

    compare_models()


if __name__ == "__main__":
    main()