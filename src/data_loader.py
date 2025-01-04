import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split
from io import BytesIO


def save_dataset_to_dataframe(output_path, sample_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.PCAM(root=dir, split="val", transform=transform, download=True)
    sample_dataset, _ = random_split(dataset, [sample_size, len(dataset) - sample_size])
    records = []

    for idx, (image, label) in enumerate(sample_dataset):
        buffer = BytesIO()
        torch.save(image, buffer)
        image_bytes = buffer.getvalue()
        records.append(
            {"image_id": f"img_{idx}", "image_tensor": image_bytes, "label": label}
        )

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow")
    print(f"Saved dataset to {output_path}")


def load_dataset_from_dataframe(file_path):
    df = pd.read_parquet(file_path, engine="pyarrow")

    def bytes_to_tensor(image_bytes):
        buffer = BytesIO(image_bytes)
        return torch.load(buffer, weights_only=True)

    df["image_tensor"] = df["image_tensor"].apply(bytes_to_tensor)
    return df


if __name__ == "__main__":
    dir = "/home/ran/datasets/"
    save_dataset_to_dataframe(output_path=f"{dir}/test.parquet", sample_size=10)
