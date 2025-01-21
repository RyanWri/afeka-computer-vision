import logging
import math
import time
import numpy as np
from src.loaders import create_data_loader
from src.models.rejectors.lof import train_lof_model
from src.models.rejectors.isolation_forest import train_isolation_forest
from src.io_utils import load_dataset_from_config, load_config


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


def train_rejection_models_from_config(config_path):
    config = load_config(config_path, add_experiment_paths=False)

    for model_config in config["rejection_models"]:
        model_name = model_config["name"].lower()
        batch_size, sample_size = (
            model_config["input"]["batch_size"],
            model_config["input"]["sample_size"],
        )

        # Load input data
        train_dataset = load_dataset_from_config(
            model_config, split="train"
        )  # Implement a function to load Patch Camelyon data
        sample_size = math.floor(len(train_dataset) * sample_size)
        train_loader = create_data_loader(
            train_dataset, sample_size=sample_size, batch_size=batch_size, num_workers=2
        )
        logging.info("start loading images")
        start = time.time()
        tensors = []
        for tensor, labels in train_loader:
            tensors.append(tensor.numpy())
        images = np.concatenate(tensors)
        logging.info(
            f"Loaded {len(images)} images in {time.time() - start:.2f} seconds"
        )

        policy = model_config["policy"]
        if model_name == "lof" and policy["enabled"]:
            train_lof_model(images, policy)

        if model_name == "isolation_forest" and policy["enabled"]:
            train_isolation_forest(images, policy)


if __name__ == "__main__":
    config_path = "train_models.yaml"
    train_rejection_models_from_config(config_path)
