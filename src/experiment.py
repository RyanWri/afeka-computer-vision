from src.models.rejection_gate import RejectionGate
from src.models.train_models import get_features


def run_experiment(config):
    """
    Run the experiment pipeline based on the loaded configuration.
    """
    features_arr, labels_arr = get_features(config, split="val")
    batch_size = config["input"]["batch_size"]

    # Initialize rejection gate
    rejection_gate = RejectionGate(config["rejection_models"]["models"])

    # Prepare results storage
    results = []

    for i in range(0, len(features_arr), batch_size):
        features = features_arr[i : i + batch_size]
        labels = labels_arr[i : i + batch_size]
        reject_scores = rejection_gate.compute_rejection_confidence(features, labels)
        results.append(reject_scores)
