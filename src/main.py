import pandas as pd
import torch
from models.baseline_cnn import BaselineCNN
from models.rejection_gate import RandomRejector
from data.loader import load_dataset_from_dataframe


def process_pipeline(df, rejection_rate=0.1):
    """
    Process images through the rejection filter and baseline CNN model.

    Args:
        df (pd.DataFrame): Input DataFrame containing image tensors and labels.
        rejection_rate (float): Rate at which the rejector rejects images.

    Returns:
        pd.DataFrame: Processed DataFrame with additional columns.
    """
    # Initialize the rejector and CNN model
    rejector = RandomRejector(rejection_rate=rejection_rate)
    cnn_model = BaselineCNN()
    cnn_model.eval()  # Ensure model is in evaluation mode

    # Results to store
    results = []

    with torch.no_grad():  # No need to compute gradients
        for _, row in df.iterrows():
            image_id = row["image_id"]
            image_tensor = row["image_tensor"]
            label = row["label"]

            # Run the rejection filter on a single image
            is_rejected = rejector.filter_single()

            if is_rejected:
                # If rejected, append result
                results.append(
                    {
                        "image_id": image_id,
                        "reject": True,
                        "model": None,
                        "label": label,
                    }
                )
            else:
                # If not rejected, pass through the baseline CNN
                prediction = cnn_model(image_tensor.unsqueeze(0)).item()
                results.append(
                    {
                        "image_id": image_id,
                        "reject": False,
                        "model": prediction,
                        "label": label,
                    }
                )

    return pd.DataFrame(results)


def load_data(parquet_file):
    """
    Load data from a Parquet file into a DataFrame.
    """
    return load_dataset_from_dataframe(parquet_file)


def save_data(df, output_file):
    """
    Save a DataFrame to a Parquet file.
    """
    df.to_parquet(output_file, engine="pyarrow")
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    # Parameters
    input_parquet = "/home/ran/datasets/test-pcam/test.parquet"
    output_parquet = "/home/ran/datasets/test-pcam/results.parquet"
    rejection_rate = 0.1

    # Load data
    df = load_data(input_parquet)

    # Process pipeline
    results_df = process_pipeline(df, rejection_rate=rejection_rate)

    # Save results
    save_data(results_df, output_parquet)
