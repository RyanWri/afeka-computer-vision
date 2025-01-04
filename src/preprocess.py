import torch
from torch.utils.data import Dataset, DataLoader

from data_loader import load_dataset_from_dataframe


class PCAMDataFrameDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_tensor = row["image_tensor"]
        label = row["label"]
        return image_tensor, torch.tensor(label, dtype=torch.long)


# Load the DataFrame
df = load_dataset_from_dataframe("/home/ran/datasets/test.parquet")

# Wrap it into a Dataset and DataLoader
dataset = PCAMDataFrameDataset(df)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Verify by iterating through the DataLoader
for batch_idx, (images, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}")
    print(f"Images: {images.shape}, Labels: {labels}")
    break  # Exit after one batch to keep it simple
