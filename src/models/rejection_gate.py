from src.models.model_factory import ModelFactory


class RejectionGate:
    def __init__(self, models_cofig: dict):
        self.rejection_models = []
        self.model_names = []
        for model_config in models_cofig:
            model = ModelFactory.create_model(
                model_config["name"], model_config["weight"]
            )
            model.load(model_config["load_path"])
            self.rejection_models.append(model)
            self.model_names.append(model_config["name"])

    def compute_rejection_confidence(self, features, labels):
        predictions = []
        for model in self.rejection_models:
            score = model.predict(features, labels)
            predictions.append(score)
        return predictions

    def get_model_names(self):
        return self.model_names
