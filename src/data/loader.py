import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
from io import BytesIO


def extract_center_patch(image, patch_size=32):
    """
    Extract the center patch of the given image tensor.
    Assumes the image is square (e.g., 96x96).
    """
    _, h, w = image.size()  # C, H, W
    start_x = (w - patch_size) // 2
    start_y = (h - patch_size) // 2
    end_x = start_x + patch_size
    end_y = start_y + patch_size
    return image[:, start_y:end_y, start_x:end_x]


def save_dataset_to_dataframe(output_path, sample_size, dataset_dir) -> bool:
    """
    Save the original 96x96 images to a Parquet file for the pipeline.
    """
    # Check if the dataset is already downloaded
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        print(f"Dataset already exists in {dataset_dir}. Skipping download.")
        return False

    # Download the dataset
    print(f"Downloading dataset to {dataset_dir}...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.PCAM(
        root=dataset_dir, split="train", transform=transform, download=True
    )

    # Limit dataset to a small sample size
    indices = list(range(sample_size))
    sample_dataset = Subset(dataset, indices)

    # Prepare records for saving
    records = []
    for idx, (image, label) in enumerate(sample_dataset):
        # Serialize the tensor to bytes
        buffer = BytesIO()
        torch.save(image, buffer)
        image_bytes = buffer.getvalue()

        # Append the record
        records.append(
            {"image_id": f"img_{idx}", "image_tensor": image_bytes, "label": label}
        )

    # Save records to Parquet
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Saved dataset to {output_path}")
    return True


def load_dataset_from_dataframe(file_path):
    df = pd.read_parquet(file_path, engine="pyarrow")

    def bytes_to_tensor(image_bytes):
        buffer = BytesIO(image_bytes)
        return torch.load(buffer)

    df["image_tensor"] = df["image_tensor"].apply(bytes_to_tensor)
    return df


if __name__ == "__main__":
    dir = "/home/ran/datasets/test-pcam/"
    parquet_path = f"{dir}/test.parquet"
    dataset_dir = f"{dir}/pcam_data"
    loaded = save_dataset_to_dataframe(
        output_path=parquet_path, sample_size=2000, dataset_dir=dataset_dir
    )
    if loaded:
        print("Dataset downloaded and saved.")
    else:
        print("Dataset already exists.")
