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


def train_rejection_models(models_config, images):
    for model_config in models_config:
        name = model_config["name"].lower()
        if name == "lof":
            train_lof_model(images, model_config)

        elif name == "isolation forest":
            train_isolation_forest(images, model_config)

        else:
            print(f"Unknown rejection model: {name}")


if __name__ == "__main__":
    config_path = "train_models.yaml"
    config = load_config(config_path, add_experiment_paths=False)

    # Load input data
    train_dataset = load_dataset_from_config(
        config, split="train"
    )  # Implement a function to load Patch Camelyon data
    sample_size = math.floor(len(train_dataset) * 0.1)
    train_loader = create_data_loader(
        train_dataset, sample_size=sample_size, batch_size=32, num_workers=2
    )
    logging.info("start loading images")
    start = time.time()
    tensors = []
    for tensor, labels in train_loader:
        tensors.append(tensor.numpy())
    images = np.concatenate(tensors)
    logging.info(f"Loaded {len(images)} images in {time.time() - start:.2f} seconds")

    # Train and save the LOF model
    logging.info("Training rejection models...")
    start = time.time()
    train_rejection_models(config["rejection_models"], images)
