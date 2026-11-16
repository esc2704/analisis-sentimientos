from experiments.common import run_experiment

if __name__ == "__main__":
    run_experiment("real", "bow")
    run_experiment("real", "word2vec")