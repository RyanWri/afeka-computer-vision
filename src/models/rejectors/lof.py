import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import joblib


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
    # Example dataset: Randomly generated images (replace with real data)
    images = np.random.rand(1000, 96, 96, 3)

    # Train and save the LOF model
    lof_data = train_lof_model(
        images=images, n_neighbors=20, contamination=0.1, save_path="./lof_model.joblib"
    )

    # Load the LOF model
    lof_data = load_lof_model("./lof_model.joblib")

    # Compute LOF scores for test images
    test_images = np.random.rand(100, 96, 96, 3)  # Replace with test data
    lof_scores = compute_lof_scores(lof_data, test_images)

    # Reject images based on a threshold
    threshold = np.percentile(lof_scores, 95)  # Top 5% are considered outliers
    rejected, scores = reject_images(lof_data, test_images, threshold)

    # Print rejection stats
    print(f"Rejection Rate: {rejected.mean() * 100:.2f}%")
