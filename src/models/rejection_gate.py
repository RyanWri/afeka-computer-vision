import torch  # For parallel predictions


class RejectionGate:
    def __init__(self, rejection_models, weights, threshold):
        """
        Initialize the rejection gate with multiple models and their weights.

        Args:
            rejection_models (list): List of rejection model instances.
            weights (list): List of weights corresponding to each model.
            threshold (float): Confidence threshold for rejection.
        """
        if len(rejection_models) != len(weights):
            raise ValueError("Number of models and weights must match.")

        self.rejection_models = rejection_models
        self.weights = torch.tensor(weights)  # Store weights as a tensor
        self.threshold = threshold

        # Validate weights sum to 1
        total_weight = self.weights.sum().item()
        if not (0.999 <= total_weight <= 1.001):  # Allow for float rounding errors
            raise ValueError("Rejection model weights must sum to 1.")

    def compute_rejection_confidence(self, input_data):
        """
        Compute overall rejection confidence based on model results and weights.

        Args:
            input_data: The input data to evaluate.

        Returns:
            float: Rejection confidence between 0 and 1.
        """
        # Predict rejection probabilities for all models
        predictions = [model.predict(input_data) for model in self.rejection_models]

        # Weighted sum of predictions
        confidence = (predictions * self.weights).sum().item()
        return confidence

    def should_reject(self, input_data):
        """
        Determine whether to reject the input based on overall confidence.

        Args:
            input_data: The input data to evaluate.

        Returns:
            bool: True if rejected, False otherwise.
        """
        image = torch.squeeze(input_data)
        confidence = self.compute_rejection_confidence(image)
        return confidence > self.threshold
