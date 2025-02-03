from src.models.rejection_gate import RejectionGate
from src.models.train_models import get_features


def run_experiment(config):
    """
    Run the experiment pipeline based on the loaded configuration.
    """
    features_arr, labels = get_features(config, split="val")

    # Initialize rejection gate
    rejection_gate = RejectionGate(config["rejection_models"]["models"])

    # Prepare results storage
    results = []

    for features in features_arr:
        reject_scores = rejection_gate.compute_rejection_confidence(features)
        results.append(sum(reject_scores))
