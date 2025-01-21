import joblib


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


# LOF Model
class LOFModel(BaseModel):
    def predict(self, image):
        """Override the predict method for LOF."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        return self.calc_score(image)

    def calc_score(self, image):
        for k, v in self.model:
            print(k)
            print(v)


# Isolation Forest Model
class IsolationForestModel(BaseModel):
    def predict(self, image):
        """Override the predict method for Isolation Forest."""
        if self.model is None:
            raise ValueError("Model not loaded. Call `load` first.")
        return self.calc_score(image)

        def calc_score(self, image):
            return 1


# Factory class
class ModelFactory:
    @staticmethod
    def create_model(model_type):
        if model_type == "lof":
            return LOFModel()
        elif model_type == "isolation_forest":
            return IsolationForestModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
