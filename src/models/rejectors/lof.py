import joblib
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import logging
import time
from io_utils import load_config, load_dataset_from_config

from loaders import create_data_loader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


def sample_data(dataset, sample_size):
    # Assuming 'dataset' can be indexed directly
    if sample_size < len(dataset):
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        sampled_data = dataset[indices]
    else:
        sampled_data = dataset  # Use the full dataset if sample_size is too large
    return sampled_data


def train_lof_model(images, n_neighbors, contamination, save_path):
    # Flatten images and standardize features
    N, H, W, C = images.shape
    features = images.reshape(N, H * W * C)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train LOF model and save
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors, contamination=contamination, novelty=True
    )
    lof.fit(features_scaled)
    joblib.dump({"model": lof, "scaler": scaler}, save_path)
    return {"model": lof, "scaler": scaler}


def load_lof_model(load_path):
    return joblib.load(load_path)


def compute_lof_scores(lof_data, images):
    # Flatten and scale images
    N, H, W, C = images.shape
    features = images.reshape(N, H * W * C)
    features_scaled = lof_data["scaler"].transform(features)

    # Compute LOF scores
    lof_model = lof_data["model"]
    return -lof_model.decision_function(
        features_scaled
    )  # Negative scores indicate outliers


def reject_images(lof_data, images, threshold):
    lof_scores = compute_lof_scores(lof_data, images)
    rejected = lof_scores >= threshold
    return rejected, lof_scores


if __name__ == "__main__":
    config_path = "train_models.yaml"
    config = load_config(config_path, add_experiment_paths=False)

    # Load input data
    train_dataset = load_dataset_from_config(
        config, split="train"
    )  # Implement a function to load Patch Camelyon data
    train_loader = create_data_loader(
        train_dataset, sample_size=32000, batch_size=32, num_workers=2
    )
    start = time.time()
    t = []
    for images, labels in train_loader:
        t.append(images.numpy())
    res = np.concatenate(t)
    logging.info(f"Loaded {len(res)} images in {time.time() - start:.2f} seconds")

    # Train and save the LOF model
    logging.info("Training LOF model...")
    model_path = "/home/ran/afeka/computer-vision/models/lof.joblib"
    start = time.time()
    lof_data = train_lof_model(
        images=images, n_neighbors=5, contamination=0.1, save_path=model_path
    )
    logging.info(f"LOF model trained and saved in {time.time() - start:.2f} seconds")

    # Load the LOF model
    lof_data = load_lof_model(model_path)

    # Compute LOF scores for test images
    test_dataset = load_dataset_from_config(
        config, split="test"
    )  # Implement a function to load Patch Camelyon data
    test_loader = create_data_loader(
        test_dataset, sample_size=3200, batch_size=32, num_workers=2
    )
    t = []
    for images, labels in test_loader:
        t.append(images.numpy())
    test_images = np.concatenate(t)
    lof_scores = compute_lof_scores(lof_data, test_images)

    # Reject images based on a threshold
    threshold = np.percentile(lof_scores, 95)  # Top 5% are considered outliers
    rejected, scores = reject_images(lof_data, test_images, threshold)

    # Print rejection stats
    print(f"Rejection Rate: {rejected.mean() * 100:.2f}%")
