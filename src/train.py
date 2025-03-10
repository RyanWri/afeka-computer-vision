import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm
from models.baseline_cnn import BaselineCNN
from torch.cuda.amp import autocast, GradScaler
from src.loaders import create_data_loader, load_dataset


def train_model(model, train_loader, test_loader, device, config):
    """
    Train the baseline CNN model.

    Args:
        model (torch.nn.Module): The CNN model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to use for training (CPU or GPU).
        config (dict): Configuration dictionary with hyperparameters and paths.

    Returns:
        torch.nn.Module: Trained model.
    """
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    num_epochs = config["epochs"]
    model.to(device)

    scaler = GradScaler()  # Scale gradients to avoid underflow
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training Phase
        model.train()
        train_loss = 0.0

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

            optimizer.zero_grad()
            with autocast():  # Use mixed precision
                outputs = model(images, return_features=False)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # Scale the loss
            scaler.step(optimizer)  # Step with scaled gradients
            scaler.update()  # Update the scaler

            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(f"Train Loss: {train_loss:.4f}")

    print("Training complete!")

    # Test the model
    test_loss, test_accuracy = evaluate_model(model, test_loader, device, criterion)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save the trained model
    torch.save(model.state_dict(), config["save_path"])
    print(f"Model saved to {config['save_path']}")


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate the model on the given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): Device to use for evaluation.
        criterion (nn.Module): Loss function.

    Returns:
        tuple: Average loss and accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Testing"):
            images, labels = images.to(device), labels.float().to(device).unsqueeze(1)

            # Forward pass
            outputs = model(images, return_features=False)  # Outputs are logits

            # Compute loss (logits are passed directly)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Convert logits to probabilities
            probabilities = torch.sigmoid(outputs)  # Apply sigmoid to logits

            # Binary predictions
            predictions = (probabilities >= 0.5).float()  # Threshold at 0.5

            # Accuracy computation
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Compute average loss and accuracy
    average_loss = total_loss / len(data_loader)
    accuracy = (correct_predictions / total_samples) * 100
    return average_loss, accuracy


def train_baseline_convolution_model(config):
    train_dataset = load_dataset(config["input"]["folder"], split="train")
    test_dataset = load_dataset(config["input"]["folder"], split="test")

    # create loader for the train
    sample_size = math.floor(len(train_dataset) * config["input"]["sample_size"])
    train_loader = create_data_loader(
        train_dataset,
        sample_size=sample_size,
        batch_size=config["input"]["batch_size"],
        num_workers=2,
    )

    sample_size = math.floor(len(test_dataset) * config["input"]["sample_size"])
    test_loader = create_data_loader(
        test_dataset,
        sample_size=sample_size,
        batch_size=config["input"]["batch_size"],
        num_workers=2,
    )

    # Initialize the model
    model = BaselineCNN()

    # Detect device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train the model
    trained_model = train_model(
        model, train_loader, test_loader, device, config["baseline_model"]["train"]
    )

    return trained_model
