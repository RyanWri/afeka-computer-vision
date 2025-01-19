import os
import torch
import yaml
from loaders import load_dataset
from models.rejection_gate import RejectionGate, RandomRejector
from models.baseline_cnn import BaselineCNN


def load_config(config_path, add_experiment_paths):
    """
    Load YAML configuration file.
    """
    # Load configuration
    config_path = get_config_path(config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config = resolve_experiment_paths(config) if add_experiment_paths else config

    return config


def load_rejection_model(rejection_model_config):
    """
    Load rejection models based on the provided configuration.
    """
    if rejection_model_config["name"] == "random_rejector":
        model = RandomRejector(rejection_model_config["path"])

    elif rejection_model_config["name"] == "dummy_rejector":
        model = RandomRejector(
            rejection_model_config["path"]
        )  # Replace with real model when available
    else:
        raise ValueError(f"Unknown rejection model: {rejection_model_config['name']}")

    return model


def load_baseline_model(model_config, device):
    """
    Load the baseline classification model.
    """
    cnn_model = BaselineCNN()
    cnn_model.load_state_dict(
        torch.load(model_config["path"], map_location=device, weights_only=True)
    )
    cnn_model.eval()
    cnn_model.to(device)
    return cnn_model


def load_dataset_from_config(config, split):
    """
    Load the dataset specified in the configuration.
    """
    input_folder = config["input"]["folder"]
    return load_dataset(input_folder, split)


def initialize_rejection_gate(rejection_models_config, rejection_gate_threshold):
    """
    Initialize the rejection gate with models and thresholds from the configuration.
    """
    rejection_models = []
    for model_config in rejection_models_config:
        rejection_models.append(load_rejection_model(model_config))

    rejection_weights = [
        model_config["weight"] for model_config in rejection_models_config
    ]

    return RejectionGate(
        rejection_models=rejection_models,
        weights=rejection_weights,
        threshold=rejection_gate_threshold,
    )


def get_config_path(relative_path):
    """
    Construct the absolute path for a configuration file.

    Args:
        relative_path (str): Path to the config file relative to the 'config' directory.

    Returns:
        str: Absolute path to the configuration file.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
    config_dir = os.path.join(
        src_dir, "config"
    )  # Relative path to the config directory
    return os.path.normpath(os.path.join(config_dir, relative_path))


def resolve_experiment_paths(config):
    """
    Dynamically resolve all paths for results and metrics based on the base root and experiment alias.

    Args:
        config (dict): Experiment configuration.

    Returns:
        dict: Updated configuration with resolved paths for results and metrics.
    """
    base_root = config["experiment"]["base_root"]
    alias = config["experiment"]["alias"]

    # Create experiment-specific directory under base_root
    experiment_root = os.path.join(base_root, alias)
    os.makedirs(experiment_root, exist_ok=True)

    # Update paths for outputs
    resolved_paths = {
        "results_path": os.path.join(experiment_root, "results.parquet"),
        "metrics_path": os.path.join(experiment_root, "metrics.json"),
    }

    # Inject resolved paths into the configuration
    config["experiment"]["results_path"] = resolved_paths["results_path"]
    config["experiment"]["metrics_path"] = resolved_paths["metrics_path"]

    return config
