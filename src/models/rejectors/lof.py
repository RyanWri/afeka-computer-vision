import joblib
import numpy as np
import logging
import time
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

    logging.info("training lof model")
    start = time.time()
    # Train LOF model and save
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1,
    )
    lof.fit(features_scaled)
    logging.info(f"completed lof training in {time.time() - start:.2f} seconds")
    joblib.dump({"model": lof, "scaler": scaler}, save_path)
    return {"model": lof, "scaler": scaler}
