import joblib
import time
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


# Base class
class BaseModel:
    def __init__(self, weight):
        self.model = None
        self.weight = weight

    def load(self, load_path):
        """Load the model from the specified path."""
        self.model = joblib.load(load_path)

    def predict(self, features):
        """Make predictions using the model."""
        raise NotImplementedError("Subclasses must implement the `predict` method.")

    def train(self, features, labels, config: dict):
        raise NotImplementedError("Subclasses must implement the `train` method.")


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


class MahalanobisModel(BaseModel):
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")

        features_scaled = self.model["scaler"].transform(features)
        distances = []

        for cls in self.model["class_means"]:
            mean = self.model["class_means"][cls]
            cov = self.model["class_covariances"][cls]
            dist = self._mahalanobis_distance(features_scaled, mean, cov)
            distances.append(dist)

        return round(min(distances) * self.weight, 5)

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
        inv_cov = np.linalg.inv(cov.covariance_)
        return np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))


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
        raise ValueError(f"Unknown model type: {model_type}")
