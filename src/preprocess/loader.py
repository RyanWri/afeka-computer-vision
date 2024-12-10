from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the dataset
dataset = load_dataset("fw407/vtab-1k_patch_camelyon")


# Analyze class distribution
def analyze_distribution(dataset_split):
    labels = [item["label"] for item in dataset_split]
    label_counts = pd.Series(labels).value_counts()
    print("\nClass distribution:")
    print(label_counts)

    # Plot distribution
    label_counts.plot(kind="bar")
    plt.title(f"Class Distribution in {dataset_split.split}")
    plt.xlabel("Label")
    plt.ylabel("Frequency")
    plt.show()


# Analyze statistics
def calculate_stats(dataset_split):
    data_stats = pd.DataFrame(dataset_split)
    print("\nDataset Statistics:")
    print(data_stats.describe())


# Run analysis on the 'train' split
print("\nAnalyzing 'train' split...")
train_split = dataset["train"]
analyze_distribution(train_split)
calculate_stats(train_split)
