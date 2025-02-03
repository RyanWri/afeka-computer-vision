import os
import torch
import yaml
from src.loaders import load_dataset
from src.models.baseline_cnn import BaselineCNN


def load_config(config_path, add_experiment_paths):
    config_path = get_config_path(config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config = resolve_experiment_paths(config) if add_experiment_paths else config

    return config


def load_baseline_model(model_config, device):
    cnn_model = BaselineCNN()
    cnn_model.load_state_dict(torch.load(model_config["path"], map_location=device))
    cnn_model.eval()
    cnn_model.to(device)
    return cnn_model


def load_dataset_from_config(config, split):
    input_folder = config["input"]["folder"]
    return load_dataset(input_folder, split)


def get_config_path(relative_path):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(src_dir, "config")
    return os.path.normpath(os.path.join(config_dir, relative_path))


def resolve_experiment_paths(config):
    base_root = config["experiment"]["base_root"]
    alias = config["experiment"]["alias"]

    experiment_root = os.path.join(base_root, alias)
    os.makedirs(experiment_root, exist_ok=True)

    resolved_paths = {
        "results_path": os.path.join(experiment_root, "results.parquet"),
        "metrics_path": os.path.join(experiment_root, "metrics.json"),
    }

    config["experiment"]["results_path"] = resolved_paths["results_path"]
    config["experiment"]["metrics_path"] = resolved_paths["metrics_path"]

    return config
