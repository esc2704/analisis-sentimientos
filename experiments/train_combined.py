from experiments.common import run_experiment

if __name__ == "__main__":
    run_experiment("combined", "bow")
    run_experiment("combined", "word2vec")