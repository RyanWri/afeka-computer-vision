import joblib
import time
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from src.loaders import extract_center_patch
from sklearn.neighbors import LocalOutlierFactor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting the training process")


# Base class
class BaseModel:
    def __init__(self):
        self.model = None

    def load(self, load_path):
        """Load the model from the specified path."""
        self.model = joblib.load(load_path)

    def predict(self, image):
        """Make predictions using the model."""
        raise NotImplementedError("Subclasses must implement the `predict` method.")

    def train(self, features, config: dict):
        raise NotImplementedError("Subclasses must implement the `train` method.")


# LOF Model
class LOFModel(BaseModel):
    def predict(self, image):
        """Override the predict method for LOF."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        return self.calc_score(image)

    def calc_score(self, image):
        center_image = extract_center_patch(image, patch_size=32)
        features = center_image.reshape(1, 32 * 32 * 3)  # Reshape to (1, features)
        features_scaled = self.model["scaler"].transform(features)
        lof_score = -self.model["model"].negative_outlier_factor_(features_scaled)
        return lof_score

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
    def predict(self, image):
        """Override the predict method for Isolation Forest."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        return self.calc_score(image)

    def calc_score(self, image):
        center_image = extract_center_patch(image, patch_size=32)
        features = center_image.reshape(1, 32 * 32 * 3)  # Reshape to (1, features)
        features_scaled = self.model["scaler"].transform(features)
        return -self.model["model"].decision_function(features_scaled)

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
    def predict(self, image):
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        return self.calc_score(image)

    def calc_score(self, features):
        features_scaled = self.model["scaler"].transform(features)
        return self.model["model"].predict(features_scaled)

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
    def create_model(model_type):
        if model_type == "lof":
            return LOFModel()
        elif model_type == "isolation_forest":
            return IsolationForestModel()
        elif model_type == "one_class_svm":
            return OneClassSVMModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
