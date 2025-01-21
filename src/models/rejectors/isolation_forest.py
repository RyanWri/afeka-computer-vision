from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


def train_isolation_forest(images, config):
    n_estimators = config["n_estimators"]
    contamination = config["contamination"]
    save_path = config["save_path"]
    N, H, W, C = images.shape
    features = images.reshape(N, H * W * C)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        n_jobs=-1,
    )
    logging.info("training isolation forest model")
    start = time.time()
    model.fit(features_scaled)
    logging.info(
        f"completed isolation forest training in {time.time() - start:.2f} seconds"
    )
    joblib.dump({"model": model, "scaler": scaler}, save_path)
    return {"model": model, "scaler": scaler}
