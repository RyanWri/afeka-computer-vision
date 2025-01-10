import pandas as pd
import torch
from tqdm import tqdm
from utils.paths import get_config_path
from utils.io import (
    load_config,
    load_dataset,
    load_baseline_model,
    initialize_rejection_gate,
)


def run_experiment(config_filename):
    """
    Run the experiment pipeline based on the loaded configuration.
    """
    # Load configuration
    config_path = get_config_path(config_filename)
    config = load_config(config_path)

    # Load dataset
    df = load_dataset(config)

    # Initialize rejection gate
    rejection_gate = initialize_rejection_gate(config)

    # Load the baseline classification model
    cnn_model = load_baseline_model(config["baseline_model"])

    # Run inference pipeline
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row["image_id"]
            image_tensor = row["image_tensor"]
            label = row["label"]

            # Apply rejection gate
            if rejection_gate.should_reject(image_tensor):
                results.append(
                    {
                        "image_id": image_id,
                        "reject": True,
                        "model": None,
                        "label": label,
                    }
                )
            else:
                # If not rejected, classify using the CNN model
                image_tensor = image_tensor.to(device).unsqueeze(0)
                prediction = cnn_model(image_tensor).item()
                results.append(
                    {
                        "image_id": image_id,
                        "reject": False,
                        "model": prediction,
                        "label": label,
                    }
                )

    # Save results
    results_path = config["experiment"]["results_path"]
    results_df = pd.DataFrame(results)
    results_df.to_parquet(results_path, engine="pyarrow")
    print(f"Experiment results saved to {results_path}")


if __name__ == "__main__":
    # Specify the configuration file path
    config_filename = "experiment_01.yaml"  # Relative to the 'config' directory
    run_experiment(config_filename)
