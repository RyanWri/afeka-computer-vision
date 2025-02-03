import torch
from src.io_utils import load_baseline_model
from src.loaders import config_to_dataloader


def run_inference(config, threshold):
    """Runs inference on a dataset and returns probabilities and predicted labels."""
    val_loader = config_to_dataloader(config, split="val")

    # Detect device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = load_baseline_model(config["baseline_model"]["load_path"], device)
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
