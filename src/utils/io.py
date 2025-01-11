import yaml
import torch
import gzip
import shutil
import os
from data.loader import load_dataset_from_dataframe
from models.rejection_gate import RejectionGate, RandomRejector
from models.baseline_cnn import BaselineCNN


def load_config(config_path):
    """
    Load YAML configuration file.
    """
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def load_rejection_model(rejection_model_config):
    """
    Load rejection models based on the provided configuration.
    """
    if rejection_model_config["name"] == "random_rejector":
        model = RandomRejector(rejection_model_config["path"])

    if rejection_model_config["name"] == "dummy_rejector":
        model = RandomRejector(
            rejection_model_config["path"]
        )  # Replace with real model when available

    else:
        raise ValueError(f"Unknown rejection model: {rejection_model_config['name']}")

    return model


def load_baseline_model(model_config):
    """
    Load the baseline classification model.
    """
    cnn_model = BaselineCNN()
    cnn_model.load_state_dict(torch.load(model_config["path"]))
    cnn_model.eval()
    return cnn_model


def load_dataset(config):
    """
    Load the dataset specified in the configuration.
    """
    input_folder = config["experiment"]["input_folder"]
    sample_size = config["experiment"]["sample_size"]

    df = load_dataset_from_dataframe(input_folder)
    return df.sample(n=sample_size)  # Limit to sample size


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


def decompress_gz_files(dataset_dir, split):
    """
    Decompress the .h5.gz files for a specific split.

    Args:
        dataset_dir (str): Path where the dataset is stored.
        split (str): Split to decompress ('train', 'val', or 'test').
    """
    required_files = [
        f"camelyonpatch_level_2_split_{split}_x.h5.gz",
        f"camelyonpatch_level_2_split_{split}_y.h5.gz",
    ]

    for file_name in required_files:
        gz_path = os.path.join(dataset_dir, file_name)
        h5_path = os.path.splitext(gz_path)[0]  # Remove the .gz extension

        # Decompress only if the uncompressed file doesn't already exist
        if not os.path.exists(h5_path):
            print(f"Decompressing {gz_path}...")
            with gzip.open(gz_path, "rb") as f_in:
                with open(h5_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Decompressed to {h5_path}")
        else:
            print(f"{h5_path} already exists. Skipping decompression.")
