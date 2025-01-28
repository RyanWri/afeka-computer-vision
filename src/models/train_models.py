import logging
import math
import time
import torch
import numpy as np
from src.feature_extractor import SingleConvFeatureExtractor
from src.loaders import create_data_loader
from src.io_utils import load_dataset_from_config, load_config
from model_factory import ModelFactory  # Import the ModelFactory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


def train_rejection_models_from_config(config_path):
    # Load the configuration
    config = load_config(config_path, add_experiment_paths=False)

    for model_config in config["rejection_models"]:
        model_name = model_config["name"].lower()
        if not model_config.get("policy", {}).get("enabled", False):
            logging.info(f"Skipping disabled model: {model_name}")
            continue
        features = get_features(model_config)
        model = ModelFactory.create_model(model_name)  # Dynamically create the model
        model.train(
            features, model_config["policy"]
        )  # Train the model using the extracted features


def get_features(model_config):
    # Extract features
    feature_extractor = SingleConvFeatureExtractor(input_channels=3, output_channels=32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size, sample_size = (
        model_config["input"]["batch_size"],
        model_config["input"]["sample_size"],
    )

    # Load the dataset
    train_dataset = load_dataset_from_config(
        model_config, split="train"
    )  # Implement a function to load Patch Camelyon data
    sample_size = math.floor(len(train_dataset) * sample_size)
    train_loader = create_data_loader(
        train_dataset, sample_size=sample_size, batch_size=batch_size, num_workers=2
    )

    logging.info("Extracting features")
    start = time.time()

    features = extract_features_from_loader(train_loader, feature_extractor, device)
    logging.info(
        f"Extracted features from {len(features)} images in {time.time() - start:.2f} seconds"
    )

    return features


def extract_features_from_loader(data_loader, feature_extractor, device):
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    all_features = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)  # Move images to the specified device
            features = feature_extractor(images)  # Extract features
            features = features.view(features.size(0), -1)  # Flatten features
            all_features.append(
                features.cpu().numpy()
            )  # Move to CPU and convert to numpy

    return np.vstack(all_features)  # Stack all features into one array


if __name__ == "__main__":
    config_path = "train_models.yaml"
    train_rejection_models_from_config(config_path)
