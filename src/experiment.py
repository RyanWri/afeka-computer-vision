import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from io_utils import (
    get_config_path,
    resolve_experiment_paths,
    load_config,
    load_dataset_from_config,
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
    config = resolve_experiment_paths(config)

    # Load test dataset only
    test_dataset = load_dataset_from_config(config, split="test")

    # Create DataLoader with shuffle and pin memory
    data_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Batch size of 1 to process individual images
        shuffle=True,  # Enable shuffling
        pin_memory=True,  # Use pinned memory for faster GPU transfers
        num_workers=4,
    )

    # Initialize rejection gate
    rejection_gate = initialize_rejection_gate(
        config["rejection_models"], config["experiment"]["rejection_gate_threshold"]
    )

    # Load the baseline convolution model
    cnn_model = load_baseline_model(config["baseline_model"])
    cnn_model.eval()  # Set the model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    # Prepare results storage
    results = []

    # Process each image in the test dataset
    for idx, (image, label) in tqdm(
        enumerate(data_loader), total=len(data_loader), desc="Processing Images"
    ):
        image = image.to(device)
        label = label.item()  # Convert label tensor to scalar

        # Apply rejection gate
        is_rejected = rejection_gate.should_reject(image)

        if not is_rejected:
            # If not rejected, classify using the CNN model
            with torch.no_grad():
                logits = cnn_model(image)  # Get logits
                probability = torch.sigmoid(logits).item()  # Apply sigmoid
                predicted_label = 1 if probability >= 0.5 else 0  # Threshold at 0.5
        else:
            # For rejected samples, no prediction
            probability = None
            predicted_label = None

        # Store results
        results.append(
            {
                "img_id": f"img_{idx}",  # Unique ID for each image
                "reject": is_rejected,
                "label": label,  # Ground truth label
                "probability": probability,  # Probability if not rejected
                "predicted_label": predicted_label,  # Predicted label if not rejected
            }
        )

    # Save results to Parquet
    results_path = config["experiment"]["results_path"]
    results_df = pd.DataFrame(results)
    results_df.to_parquet(results_path, engine="pyarrow")
    print(f"Experiment results saved to {results_path}")


if __name__ == "__main__":
    # Specify the configuration file path
    config_filename = "experiment_projlab.yaml"  # Relative to the 'config' directory
    run_experiment(config_filename)
