import logging
import time
import torch
import numpy as np
from src.loaders import create_data_loader, load_dataset
from src.io_utils import load_config
from src.models.baseline_cnn import BaselineCNN
from src.models.model_factory import ModelFactory

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


def train_rejection_models_from_config(config_path):
    # Load the configuration
    config = load_config(config_path, add_experiment_paths=False)
    features, labels = get_features(config, split="train")

    for model_config in config["rejection_models"]:
        model_name = model_config["name"].lower()
        if not model_config.get("policy", {}).get("enabled", False):
            logging.info(f"Skipping disabled model: {model_name}")
            continue
        # Dynamically create the model
        model = ModelFactory.create_model(model_name, weight=model_config["weight"])
        # Train the model using the extracted features
        model.train(features, labels, model_config["policy"])


def get_features(config, split):
    """Extracts features using the trained BaselineCNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    model.load_state_dict(
        torch.load(config["baseline_model"]["save_path"], map_location=device)
    )
    model.eval()

    input_folder, batch_size, sample_size = config["input"].values()
    train_dataset = load_dataset(input_folder, split="train")
    sample_size = int(len(train_dataset) * sample_size)
    train_loader = create_data_loader(
        train_dataset, sample_size=sample_size, batch_size=batch_size, num_workers=2
    )

    logging.info("Extracting features with BaselineCNN")
    start = time.time()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in train_loader:
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
