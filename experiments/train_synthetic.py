from experiments.common import run_experiment

if __name__ == "__main__":
    run_experiment("synthetic", "bow")
    run_experiment("synthetic", "word2vec")