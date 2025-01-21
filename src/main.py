import pandas as pd
import json
import os
from src.experiment import run_experiment
from src.io_utils import load_config
from src.metrics import summarize_metrics
from src.train import train_baseline_convolution_model


if __name__ == "__main__":
    # Specify the configuration file path
    experiment_name = "experiment_projlab"
    config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    config = load_config(config_filename, add_experiment_paths=True)

    baseline_cnn_config = config["baseline_model"]
    if baseline_cnn_config["enabled"] and not os.path.exists(
        baseline_cnn_config["path"]
    ):
        cnn_model = train_baseline_convolution_model(baseline_cnn_config["path"])
        print("traine basedline convloution completed")
    else:
        print("no need to train cnn baseline, it exist")

    run_experiment(config)
    results_df = pd.read_parquet(config["experiment"]["results_path"])
    metrics = summarize_metrics(results_df)
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(config["experiment"]["metrics_path"], "w") as f:
        json.dump(metrics, f)
