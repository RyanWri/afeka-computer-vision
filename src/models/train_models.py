import logging
import time
import torch
import numpy as np
from src.loaders import config_to_dataloader
from src.io_utils import load_baseline_model
from src.models.model_factory import ModelFactory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_rejection_models(config):
    train_loader = config_to_dataloader(config, split="train")
    features, labels = get_features(
        model_load_path=config["baseline_model"]["load_path"], data_loader=train_loader
    )

    for model_config in config["rejection_models"]["models"]:
        model_name = model_config["name"].lower()
        # Dynamically create the model
        model = ModelFactory.create_model(model_name, weight=model_config["weight"])
        # Train the model using the extracted features
        model.train(features, labels, model_config["policy"])


def get_features(model_load_path, data_loader):
    """Extracts features using the trained BaselineCNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    model = load_baseline_model(model_load_path, device)
    model.eval()
    model.to(device)

    logging.info("Extracting features with BaselineCNN")
    start = time.time()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model(images, return_features=True)  # Extract features
            features = features.view(features.size(0), -1).cpu().numpy()
            all_features.append(features)
            all_labels.append(labels.cpu().numpy())

    features_array = np.vstack(all_features)
    labels_array = np.hstack(all_labels)

    logging.info(
        f"Extracted features from {len(features_array)} images in {time.time() - start:.2f} seconds"
    )
    return features_array, labels_array
