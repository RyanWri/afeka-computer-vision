from src.inference import run_inference
from src.io_utils import dict_to_dataframe, load_config
from src.models.train_models import train_rejection_models_from_config
from src.train import train_baseline_convolution_model


if __name__ == "__main__":
    config_path = "full_project.yaml"
    config = load_config(config_path)

    # train baseline model
    if config["baseline_model"]["train"]["enabled"]:
        train_baseline_convolution_model(config)

    # run inference for baseline
    if config["baseline_model"]["inference"]["enabled"]:
        results = run_inference(config, threshold=0.5)
        dict_to_dataframe(
            results, config["baseline_model"]["inference"]["original_results"]
        )

    # train rejection models
    if config["rejection_models"]["enabled"]:
        train_rejection_models_from_config(config_path)

    # Run Experiment if enabled
    # experiment_name = "experiment_01"
    # config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    # config = load_config(config_filename, add_experiment_paths=True)
    # run_experiment(config)
