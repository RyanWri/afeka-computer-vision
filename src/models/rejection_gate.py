from src.models.model_factory import ModelFactory


class RejectionGate:
    def __init__(self, models_cofig: dict):
        self.rejection_models = []
        for model_config in models_cofig:
            model = ModelFactory.create_model(
                model_config["name"], model_config["weight"]
            )
            model.load(model_config["path"])
            self.rejection_models.append(model)

    def compute_rejection_confidence(self, input_data):
        predictions = [model.predict(input_data) for model in self.rejection_models]
        return sum(predictions)
