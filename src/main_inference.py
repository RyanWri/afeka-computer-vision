import numpy as np
from src.inference import run_inference
from src.io_utils import dict_to_dataframe, load_config
from src.models.rejection_gate import RejectionGate
from src.models.train_models import get_features
from collections import defaultdict


def run_experiment(config):
    """
    Run the experiment pipeline based on the loaded configuration.
    """
    features_arr, labels_arr = get_features(config, split="val")
    batch_size = config["input"]["batch_size"]

    # Initialize rejection gate
    rejection_gate = RejectionGate(config["rejection_models"]["models"])

    # Prepare results storage
    results = defaultdict(list)
    model_names = rejection_gate.get_model_names()

    for i in range(0, len(features_arr), batch_size):
        features = features_arr[i : i + batch_size]
        labels = labels_arr[i : i + batch_size]
        reject_scores = rejection_gate.compute_rejection_confidence(features, labels)
        for model_name, predictions in zip(model_names, reject_scores):
            results[model_name].append(predictions)

    return {
        key: np.concatenate(values, axis=0).tolist() for key, values in results.items()
    }


if __name__ == "__main__":
    config_path = "rejection_gate.yaml"
    config = load_config(config_path)

    # run inference for baseline
    if config["baseline_model"]["enabled"]:
        results = run_inference(config, threshold=0.5)
        dict_to_dataframe(
            results, config["baseline_model"]["inference"]["original_results"]
        )

    # Run Experiment if enabled
    results = run_experiment(config)
    dict_to_dataframe(results, config["experiment"]["save_path"])
