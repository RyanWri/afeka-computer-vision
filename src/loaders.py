import os
import h5py
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
import numpy as np
from torchvision import datasets, transforms


class PCamHDF5Dataset(Dataset):
    """
    A PyTorch Dataset to load PCam data directly from HDF5 files.
    """

    def __init__(self, h5_file_x_path, h5_file_y_path, transform, reduce_to_center):
        """
        Args:
            h5_file_x_path (str): Path to the HDF5 file containing image data.
            h5_file_y_path (str): Path to the HDF5 file containing label data.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.h5_file_x = h5py.File(h5_file_x_path, "r")
        self.h5_file_y = h5py.File(h5_file_y_path, "r")
        self.images = self.h5_file_x["x"]  # Update key if using another split
        self.labels = self.h5_file_y["y"]  # Update key if using another split
        self.transform = transform
        self.reduce_to_center = reduce_to_center

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        # Load image and label
        image = self.images[idx]  # NumPy array

        if self.reduce_to_center:
            image = extract_center_patch(image, patch_size=32)

        # Convert image to PyTorch tensor and normalize to [0, 1]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(
            self.labels[idx], dtype=torch.float32
        ).squeeze()  # Remove extra dimensions

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def close(self):
        """
        Close the HDF5 files when done.
        """
        self.h5_file_x.close()
        self.h5_file_y.close()


def ensure_split_exists(dataset_dir, split):
    """
    Ensure that the specified split of the PCam dataset exists.

    Args:
        dataset_dir (str): Path where the dataset splits are stored.
        split (str): Split to check ('train', 'val', or 'test').
    """
    split_name = "valid" if split == "val" else split
    required_files = [
        f"camelyonpatch_level_2_split_{split_name}_x.h5.gz",
        f"camelyonpatch_level_2_split_{split_name}_y.h5.gz",
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


def load_dataset(dataset_dir, split, reduce_to_center):
    if split == "train":
        # dataset link: https://github.com/basveeling/pcam
        message = """Please ensure train split is downloaded. 
        If not You must download train split manually due to size and decompression, 
        the download link is in the code"""
        print(message)

    if split in ["val", "test"]:
        ensure_split_exists(dataset_dir, split)

    # fix val name edge case
    split = "valid" if split == "val" else split
    normalize_transform = transforms.Normalize(mean=[0.5], std=[0.5])
    dataset = PCamHDF5Dataset(
        h5_file_x_path=os.path.join(
            dataset_dir, "pcam", f"camelyonpatch_level_2_split_{split}_x.h5"
        ),
        h5_file_y_path=os.path.join(
            dataset_dir, "pcam", f"camelyonpatch_level_2_split_{split}_y.h5"
        ),
        transform=normalize_transform,
        reduce_to_center=reduce_to_center,
    )
    return dataset


def create_data_loader(dataset, sample_size, batch_size, num_workers):
    if sample_size < len(dataset):
        indices = np.random.choice(len(dataset), sample_size, replace=False)
    else:
        indices = np.arange(len(dataset))

    sampler = SubsetRandomSampler(indices)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )
    return data_loader


def extract_center_patch(image, patch_size):
    """
    Extracts the center patch of a specified size from a given image.
    Assumes the image is square.

    Args:
        image (array): The input image array (H, W, C).
        patch_size (int): The size of the center patch to extract.

    Returns:
        array: The extracted center patch.
    """
    center_start = (image.shape[0] - patch_size) // 2
    center_end = center_start + patch_size
    return image[center_start:center_end, center_start:center_end, :]
