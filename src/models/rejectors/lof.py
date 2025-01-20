import joblib
import numpy as np
import logging
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

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


def train_lof_model(images, config):
    n_neighbors = config["n_neighbors"]
    contamination = config["contamination"]
    save_path = config["save_path"]
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
