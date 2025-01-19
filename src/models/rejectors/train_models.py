import logging
import time
import numpy as np
from models.rejectors.lof import train_lof_model
from models.rejectors.isolation_forest import train_isolation_forest
import torch
from train import (
    train_baseline_convolution_model,
)  # Assumes your CNN training function is here
from io_utils import load_dataset_from_config, load_config
from torch.utils.data import DataLoader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


def train_rejection_models(config, train_loader):
    images = []
    star = time.time()
    for batch in train_loader:
        inputs, _ = batch
        images.extend(
            inputs.numpy()
        )  # Assuming the images are tensors and need to be converted to NumPy arrays

    # Convert list to a NumPy array
    images = np.array(images)
    logging.info(f"Loaded {len(images)} images in {time.time() - star:.2f} seconds")

    for model_config in config["rejection_models"]:
        name = model_config["name"].lower()
        save_path = model_config["save_path"]

        if name == "lof":
            train_lof_model(
                images=images,
                n_neighbors=model_config["n_neighbors"],
                contamination=model_config["contamination"],
                save_path=save_path,
            )
            print(f"Trained and saved LOF model to {save_path}")

        elif name == "isolation forest":
            train_isolation_forest(
                images=images,
                n_estimators=model_config["n_estimators"],
                max_samples=model_config["max_samples"],
                contamination=model_config["contamination"],
                save_path=save_path,
            )
            print(f"Trained and saved Isolation Forest model to {save_path}")

        else:
            print(f"Unknown rejection model: {name}")


def train_baseline_model(config, train_loader, test_loader, device):
    baseline_config = config["baseline_model"]
    if baseline_config["enabled"]:
        model = train_baseline_convolution_model(
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            save_path=baseline_config["save_path"],
        )
        print(f"Trained and saved baseline CNN to {baseline_config['save_path']}")
    else:
        print("Baseline CNN training skipped (enabled=false)")


def main(config_path):
    # Load configuration
    config = load_config(config_path, add_experiment_paths=False)

    # Load input data
    train_dataset = load_dataset_from_config(
        config, split="train"
    )  # Implement a function to load Patch Camelyon data
    test_dataset = load_dataset_from_config(
        config, split="test"
    )  # Implement a function to load Patch Camelyon data

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train rejection models
    train_rejection_models(config, train_loader)

    # Train baseline model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_baseline_model(config, train_loader, test_loader, device)


if __name__ == "__main__":
    config_path = "train_models.yaml"
    main(config_path)
