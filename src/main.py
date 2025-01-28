import pandas as pd
from src.experiment import run_experiment
from src.io_utils import load_config


if __name__ == "__main__":
    # Specify the configuration file path
    experiment_name = "experiment_01"
    config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    config = load_config(config_filename, add_experiment_paths=True)

    run_experiment(config)
    results_path = config["experiment"]["results_path"]
    results_df = pd.read_parquet(results_path)
    print("Your experiment results saved at ", results_path)
