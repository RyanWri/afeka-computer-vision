import os
import h5py
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class PCamHDF5Dataset(Dataset):
    """
    A PyTorch Dataset to load PCam data directly from HDF5 files.
    """

    def __init__(self, h5_file_path, split, transform=None):
        """
        Args:
            h5_file_path (str): Path to the HDF5 file (e.g., downloaded by torchvision.datasets.PCAM).
            split (str): Dataset split to use ('train', 'val', or 'test').
            transform (callable, optional): Transformations to apply to the images.
        """
        self.h5_file_path = h5_file_path
        self.split = split
        self.transform = transform

        # Open the HDF5 file
        self.h5_file = h5py.File(h5_file_path, "r")
        self.images = self.h5_file[f"{split}_x"]
        self.labels = self.h5_file[f"{split}_y"]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Load image and label
        image = self.images[idx]  # NumPy array
        label = self.labels[idx]  # Binary label (0 or 1)

        # Convert image to PyTorch tensor and normalize to [0, 1]
        image = (
            torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        )  # C, H, W

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def close(self):
        """
        Close the HDF5 file when done.
        """
        self.h5_file.close()


def ensure_split_exists(dataset_dir, split):
    """
    Ensure that the specified split of the PCam dataset exists.

    Args:
        dataset_dir (str): Path where the dataset splits are stored.
        split (str): Split to check ('train', 'val', or 'test').
    """
    split = "valid" if split == "val" else split
    required_files = [
        f"camelyonpatch_level_2_split_{split}_x.h5.gz",
        f"camelyonpatch_level_2_split_{split}_y.h5.gz",
    ]

    # Check if required files exist
    missing_files = [
        f
        for f in required_files
        if not os.path.exists(os.path.join(dataset_dir, "pcam", f))
    ]
    if missing_files:
        print(f"Missing files for {split} split. Downloading...")
        datasets.PCAM(root=dataset_dir, split=split, download=True)
        print(f"Downloaded {split} split.")
    else:
        print(f"All files for {split} split are present.")


# Example usage
if __name__ == "__main__":
    dataset_dir = "/home/ran/datasets/pcam"  # Path where PCam dataset is stored

    # You must download train split manually due to size and decompression
    # link: https://github.com/basveeling/pcam
    for split in ["val", "test"]:
        ensure_split_exists(dataset_dir, split)

    # Define normalization transform
    normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])

    # Create dataset and DataLoader for training split
    train_dataset = PCamHDF5Dataset(
        h5_file_path=os.path.join(
            dataset_dir, "camelyonpatch_level_2_split_train_x.h5"
        ),
        split="train",
        transform=normalize_transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Iterate through the DataLoader
    for images, labels in train_loader:
        print(f"Batch of images: {images.shape}, Batch of labels: {labels.shape}")
        break

    # Close the dataset
    train_dataset.close()
