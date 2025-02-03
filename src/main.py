import pandas as pd
from src.experiment import run_experiment
from src.io_utils import load_config
from src.models.train_models import train_rejection_models_from_config
from src.train import train_baseline_convolution_model


if __name__ == "__main__":
    config_path = "train_models.yaml"
    config = load_config(config_path, add_experiment_paths=False)

    # train baseline model if enabled
    train_baseline_convolution_model(config)

    # train rejection models if enabled
    train_rejection_models_from_config(config_path)

    # Run Experiment if enabled
    experiment_name = "experiment_01"
    config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    config = load_config(config_filename, add_experiment_paths=True)
    run_experiment(config)

    # save results of experiment
    results_path = config["experiment"]["results_path"]
    results_df = pd.read_parquet(results_path)
    print("Your experiment results saved at ", results_path)
