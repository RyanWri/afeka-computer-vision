import os
import numpy as np
from src.inference import run_inference
from src.io_utils import dict_to_dataframe, load_config
from src.loaders import config_to_dataloader
from src.models.rejection_gate import RejectionGate
from src.models.train_models import get_features
from collections import defaultdict
from datetime import datetime


def run_experiment(features_arr, labels_arr, batch_size, models_config):
    """
    Run the experiment pipeline based on the loaded configuration.
    """
    # Initialize rejection gate
    rejection_gate = RejectionGate(models_config)

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
    config_path = "rejection_gate_local.yaml"
    config = load_config(config_path)
    val_loader = config_to_dataloader(config, split="val")
    batch_size = config["input"]["batch_size"]
    baseline_model_path = config["baseline_model"]["load_path"]

    # create experiment folder
    id = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_folder = os.path.join(config["output"]["folder"], id)
    os.makedirs(experiment_folder, exist_ok=True)

    # run inference for baseline
    results = run_inference(val_loader, baseline_model_path, threshold=0.5)
    original_file_path = os.path.join(
        experiment_folder, config["baseline_model"]["original_results"]
    )
    dict_to_dataframe(results, original_file_path)

    # extract features and labels for all experiments
    features_arr, labels_arr = get_features(baseline_model_path, val_loader)

    # Run Experiment if enabled
    for experiment in config["experiments"]:
        results = run_experiment(
            features_arr,
            labels_arr,
            batch_size,
            experiment["rejection_models"]["models"],
        )
        result_file_path = os.path.join(experiment_folder, experiment["save_path"])
        dict_to_dataframe(results, result_file_path)
