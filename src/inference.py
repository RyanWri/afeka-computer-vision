import torch
import math
from src.io_utils import load_baseline_model
from src.loaders import create_data_loader, load_dataset


def prepare_validation(config):
    val_dataset = load_dataset(config["input"]["folder"], split="val")

    # create loader for the train
    sample_size = math.floor(len(val_dataset) * 1)
    val_loader = create_data_loader(
        val_dataset,
        sample_size=sample_size,
        batch_size=config["input"]["batch_size"],
        num_workers=2,
    )

    return val_loader


def run_inference(config, threshold):
    """Runs inference on a dataset and returns probabilities and predicted labels."""
    val_loader = prepare_validation(config)

    # Detect device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = load_baseline_model(
        config["baseline_model"]["inference"]["load_path"], device
    )
    model.eval()
    model.to(device)

    results = {"probability": [], "prediction": [], "true_label": []}

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)

            # Forward pass
            outputs = model(images, return_features=False)  # Get logits

            # Convert logits to probabilities
            probabilities = (
                torch.sigmoid(outputs).cpu().numpy().flatten()
            )  # Move to CPU, convert to NumPy
            predicted_labels = (probabilities >= threshold).astype(
                int
            )  # Binarize predictions
            true_labels = labels.cpu().numpy().flatten()

            # Store results in dictionary
            results["probability"].extend(probabilities.tolist())
            results["prediction"].extend(predicted_labels.tolist())
            results["true_label"].extend(true_labels.tolist())

    return results
