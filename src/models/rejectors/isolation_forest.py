from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib


def train_isolation_forest(images, config):
    n_estimators = config["n_estimators"]
    max_samples = config["max_samples"]
    contamination = config["contamination"]
    save_path = config["save_path"]
    N, H, W, C = images.shape
    features = images.reshape(N, H * W * C)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features_scaled)

    joblib.dump({"model": model, "scaler": scaler}, save_path)
    return {"model": model, "scaler": scaler}


def load_isolation_forest(load_path):
    return joblib.load(load_path)


def compute_isolation_scores(model_data, images):
    N, H, W, C = images.shape
    features = images.reshape(N, H * W * C)
    features_scaled = model_data["scaler"].transform(features)

    model = model_data["model"]
    return -model.decision_function(features_scaled)


def reject_images(model_data, images, threshold):
    scores = compute_isolation_scores(model_data, images)
    rejected = scores >= threshold
    return rejected, scores
