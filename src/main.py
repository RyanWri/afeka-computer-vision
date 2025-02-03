from src.inference import run_inference
from src.io_utils import dict_to_dataframe, load_config
from src.models.train_models import train_rejection_models_from_config
from src.train import train_baseline_convolution_model


if __name__ == "__main__":
    config_path = "full_project.yaml"
    config = load_config(config_path, add_experiment_paths=False)

    # train baseline model if enabled
    if config["baseline_model"]["enabled"]:
        train_baseline_convolution_model(config)

    # train rejection models if enabled
    if config["rejection_models"]["enabled"]:
        train_rejection_models_from_config(config_path)

    # run inference for baseline
    results = run_inference(config, threshold=0.5)
    dict_to_dataframe(
        results, config["baseline_model"]["inference"]["original_results"]
    )

    # Run Experiment if enabled
    # experiment_name = "experiment_01"
    # config_filename = f"{experiment_name}.yaml"  # Relative to the 'config' directory
    # config = load_config(config_filename, add_experiment_paths=True)
    # run_experiment(config)
