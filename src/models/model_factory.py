import joblib
import time
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
import torch

from src.io_utils import load_baseline_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Base class
class BaseModel:
    def __init__(self, weight):
        self.model = None
        self.weight = weight

    def load(self, load_path):
        """Load the model from the specified path."""
        self.model = joblib.load(load_path)

    def predict(self, features, labels=None):
        """Make predictions using the model."""
        raise NotImplementedError("Subclasses must implement the `predict` method.")

    def train(self, features, labels, config: dict):
        raise NotImplementedError("Subclasses must implement the `train` method.")


class MahalanobisModel(BaseModel):
    def predict(self, features, labels):
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")

        if labels is None:
            raise ValueError("Mahalanobis Model requires labels for inference.")

        features_scaled = self.model["scaler"].transform(features)
        distances = []

        for cls in self.model["class_means"]:
            mean = self.model["class_means"][cls]
            cov = self.model["class_covariances"][cls]
            dist = self._mahalanobis_distance(features_scaled, mean, cov)
            distances.append(dist)

        distances = np.nan_to_num(distances, nan=np.nanmean(distances))
        distances = np.min(distances, axis=0)
        min_val = np.min(distances)
        max_val = np.max(distances)

        if max_val - min_val == 0:
            return np.zeros_like(distances)

        normalized_scores = (distances - min_val) / (max_val - min_val)
        batch_scores = np.mean(normalized_scores, axis=0) * self.weight
        # Shape: (batch_size, 1)
        return batch_scores

    def train(self, features, labels, config: dict):
        save_path = config["save_path"]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        logging.info("Extracting class-wise means and covariances")
        start = time.time()

        class_means = {}
        class_covariances = {}

        unique_classes = np.unique(labels)
        for cls in unique_classes:
            cls_features = features_scaled[labels == cls]
            mean_vec = np.mean(cls_features, axis=0)
            cov_mat = EmpiricalCovariance().fit(cls_features)
            class_means[cls] = mean_vec
            class_covariances[cls] = cov_mat

        logging.info(f"Completed training in {time.time() - start:.2f} seconds")
        joblib.dump(
            {
                "class_means": class_means,
                "class_covariances": class_covariances,
                "scaler": scaler,
            },
            save_path,
        )
        logging.info(f"Mahalanobis model saved at {save_path}")

    def _mahalanobis_distance(self, x, mean, cov):
        diff = x - mean
        inv_cov = np.linalg.pinv(cov.covariance_)
        return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))


class KNNModel(BaseModel):
    def predict(self, features, labels=None):
        """Predict using trained k-NN model."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")

        features_scaled = self.model["scaler"].transform(features)
        # Get distances to k nearest neighbors
        distances, _ = self.model["model"].kneighbors(features_scaled)
        mean_distances = np.mean(distances, axis=1)  # Average distance per sample

        # Normalize distances to [0,1]
        min_val = np.min(mean_distances)
        max_val = np.max(mean_distances)

        if max_val - min_val == 0:  # Avoid division by zero
            return np.zeros_like(mean_distances)

        normalized_scores = (mean_distances - min_val) / (max_val - min_val)
        return normalized_scores * self.weight

    def train(self, features, labels, config: dict):
        """Train k-NN on extracted features."""
        n_neighbors = config.get("n_neighbors", 5)
        save_path = config["save_path"]

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        logging.info(f"Training k-NN with k={n_neighbors}")
        start = time.time()

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(features_scaled, labels)

        logging.info(f"Completed k-NN training in {time.time() - start:.2f} seconds")

        joblib.dump({"model": knn, "scaler": scaler}, save_path)
        logging.info(f"k-NN model saved at {save_path}")


class MarginModel(BaseModel):
    def __init__(self, weight):
        super().__init__(weight)
        self.fc_layer = None  # Store only the last fully connected layer
        self.boundary = 0.5  # default boundary

    def load(self, model_path):
        """Loads only the last fully connected layer from the trained CNN."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_baseline_model(model_path, device)
        self.fc_layer = model.fc  # Extract only the final layer
        self.fc_layer.eval()  # Set to evaluation mode
        logging.info(f"Loaded fully connected layer from {model_path}")

    def predict(self, features, labels=None):
        """Computes the confidence margin |probability - 0.5|, scaled to [0,1]."""
        if self.fc_layer is None:
            raise ValueError("Fully connected layer not loaded. Call `load()` first.")

        # Run inference on extracted features using only the last FC layer
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32)
            logits = self.fc_layer(features_tensor)
            probabilities = (
                torch.sigmoid(logits).cpu().numpy().flatten()
            )  # Using torch.sigmoid()!

        # Compute margin |probability - 0.5| and scale by 2 to ensure [0,1] range
        margin_scores = np.abs(probabilities - self.boundary) * 2
        return margin_scores * self.weight

    def train(self, features, labels, config: dict):
        """No training required for Margin Model—just logs a message."""
        self.boundary = config["boundary"]
        logging.info("Margin Model does not require training.")


# LOF Model
class LOFModel(BaseModel):
    def predict(self, features):
        """Override the predict method for LOF."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        features_scaled = self.model["scaler"].transform(features)
        lof_score = -self.model["model"].negative_outlier_factor_(features_scaled)
        return round(lof_score * self.weight, 5)

    def train(self, features, config: dict):
        n_neighbors = config["n_neighbors"]
        contamination = config["contamination"]
        save_path = config["save_path"]
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
        logging.info(f"LOF model saved at {save_path}")


# Isolation Forest Model
class IsolationForestModel(BaseModel):
    def predict(self, features):
        """Override the predict method for Isolation Forest."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        features_scaled = self.model["scaler"].transform(features)
        if_score = -self.model["model"].decision_function(features_scaled)
        return round(if_score * self.weight, 5)

    def train(self, features, config: dict):
        n_estimators = config["n_estimators"]
        contamination = config["contamination"]
        save_path = config["save_path"]
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
        logging.info(f"isolation forest model saved at {save_path}")


class OneClassSVMModel(BaseModel):
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        features = features.reshape(1, -1)
        features_scaled = self.model["scaler"].transform(features)
        osvm_score = float(self.model["model"].predict(features_scaled))
        return round(osvm_score * self.weight, 5)

    def train(self, features, config: dict):
        nu = config["nu"]
        kernel = config["kernel"]
        gamma = config["gamma"]
        save_path = config["save_path"]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        logging.info("training one class svm model")
        start = time.time()
        model.fit(features_scaled)
        logging.info(
            f"completed one class svm training in {time.time() - start:.2f} seconds"
        )
        joblib.dump({"model": model, "scaler": scaler}, save_path)
        logging.info(f"one class svm model saved at {save_path}")


# Factory class
class ModelFactory:
    @staticmethod
    def create_model(model_type, weight):
        if model_type == "lof":
            return LOFModel(weight)
        if model_type == "isolation_forest":
            return IsolationForestModel(weight)
        if model_type == "one-class-svm":
            return OneClassSVMModel(weight)
        if model_type == "mahalanobis":
            return MahalanobisModel(weight)
        if model_type == "knn":
            return KNNModel(weight)
        if model_type == "margin":
            return MarginModel(weight)
        raise ValueError(f"Unknown model type: {model_type}")
