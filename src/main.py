import pandas as pd
import json
import os
from experiment import run_experiment
from io_utils import load_config
from metrics import summarize_metrics
from train import train_baseline_convolution_model


if __name__ == "__main__":
    experiment_name = "experiment_baseline_no_rejection"
    # Specify the configuration file path
    config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    config = load_config(config_filename)

    should_train_model = False
    if should_train_model or not os.path.exists(config["baseline_model"]["path"]):
        cnn_model = train_baseline_convolution_model(config["baseline_model"]["path"])
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
