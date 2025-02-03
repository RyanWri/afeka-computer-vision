from src.io_utils import load_config
from src.models.train_models import train_rejection_models
from src.train import train_baseline_convolution_model


if __name__ == "__main__":
    config_path = "train_models.yaml"
    config = load_config(config_path)

    # train baseline model
    train_baseline_convolution_model(config)

    # train rejection models
    train_rejection_models(config)

    # Run Experiment if enabled
    # experiment_name = "experiment_01"
    # config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    # config = load_config(config_filename, add_experiment_paths=True)
    # run_experiment(config)
