import os
import torch
import yaml
import pandas as pd
from src.models.baseline_cnn import BaselineCNN


def load_config(config_path):
    config_path = get_config_path(config_path)
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def load_baseline_model(model_path: str, device):
    cnn_model = BaselineCNN()
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    return cnn_model


def get_config_path(relative_path):
    src_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(src_dir, "config")
    return os.path.normpath(os.path.join(config_dir, relative_path))


def dict_to_dataframe(d: dict, path: str) -> None:
    df = pd.DataFrame(d)
    df.to_csv(path)
