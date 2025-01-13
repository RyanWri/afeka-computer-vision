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

    # Load dataset in splits
    train_dataset = load_dataset_from_config(config, split="train")
    test_dataset = load_dataset_from_config(config, split="test")
    val_dataset = load_dataset_from_config(config, split="val")

    # create loader for the train
    train_loader = DataLoader(
        train_dataset, batch_size=config["experiment"]["batch_size"], shuffle=False
    )
    for x in train_loader:
        print(x)

    # Initialize rejection gate
    rejection_gate = initialize_rejection_gate(
        config["rejection_models"], config["experiment"]["rejection_gate_threshold"]
    )

    # Load the baseline classification model
    cnn_model = load_baseline_model(config["baseline_model"])

    # Run inference pipeline
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)
    with torch.no_grad():
        for batch in tqdm(train_loader, total=len(train_loader)):
            images, labels = batch  # Unpack batch
            images, labels = images.to(device), labels.to(device)

            for i in range(len(images)):
                image_tensor = images[i]
                label = labels[i].item()

                # Apply rejection gate
                if rejection_gate.should_reject(image_tensor):
                    results.append(
                        {
                            "image_id": f"img_{i}",  # Use a placeholder or unique ID if available
                            "reject": True,
                            "model": None,
                            "label": label,
                        }
                    )
                else:
                    # If not rejected, classify using the CNN model
                    prediction = cnn_model(image_tensor.unsqueeze(0)).item()
                    results.append(
                        {
                            "image_id": f"img_{i}",  # Use a placeholder or unique ID if available
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
    config_filename = "experiment_projlab.yaml"  # Relative to the 'config' directory
    run_experiment(config_filename)
