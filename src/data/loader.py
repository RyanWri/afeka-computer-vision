# Data loading and augmentation logic
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load the dataset
pcam_source = "dpdl-benchmark/patch_camelyon"
import logging

logging.basicConfig(level=logging.DEBUG)
dataset = load_dataset(pcam_source, split="validation")

# Extract the first image and label
first_image = dataset[0]["image"]
first_label = dataset[0]["label"]

# Display the first image
plt.imshow(first_image)
plt.title(f"Label: {first_label}")
plt.axis("off")
plt.show()

# Calculate label distribution
labels = [sample["label"] for sample in dataset]
unique_labels, counts = zip(*[(label, labels.count(label)) for label in set(labels)])

# Plot label distribution
plt.bar(unique_labels, counts)
plt.title("Label Distribution in Train Dataset")
plt.xlabel("Labels")
plt.ylabel("Counts")
plt.show()
