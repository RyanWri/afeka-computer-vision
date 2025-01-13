import h5py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(results_path):
    """
    Calculate evaluation metrics from the results dataset.
    Args:
        results_path (str): Path to the HDF5 file containing the results.
    Returns:
        dict: Dictionary containing rejection rate and classification metrics.
    """
    with h5py.File(results_path, "r") as f:
        labels = f["label"][:]
        if f.get("rejection") is None:
            rejection_flags = [0] * len(labels)  # All samples are non-rejected in this casef
        else:
            rejection_flags = f["rejection"][:]
        predictions = f["prediction"][:]  # Assumes predictions exist for non-rejected samples

    # Calculate rejection rate
    rejection_rate = rejection_flags.mean()

    # Non-rejected indices
    non_rejected_indices = ~rejection_flags
    covered_labels = labels[non_rejected_indices]
    covered_predictions = predictions[non_rejected_indices]

    # Classification metrics for non-rejected samples
    accuracy = accuracy_score(covered_labels, covered_predictions > 0.5)
    precision = precision_score(covered_labels, covered_predictions > 0.5)
    recall = recall_score(covered_labels, covered_predictions > 0.5)
    f1 = f1_score(covered_labels, covered_predictions > 0.5)
    auc = roc_auc_score(covered_labels, covered_predictions)

    return {
        "rejection_rate": rejection_rate,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
    }


def compare_metrics(metrics1, metrics2, name1, name2):
    """
    Compare evaluation metrics between two results datasets.
    Args:
        metrics1 (dict): Metrics from the first dataset.
        metrics2 (dict): Metrics from the second dataset.
        name1 (str): Name of the first dataset.
        name2 (str): Name of the second dataset.
    """
    print(f"\nComparison between {name1} and {name2}:")
    for key in metrics1.keys():
        print(f"{key.capitalize()}:")
        print(f"  {name1}: {metrics1[key]:.4f}")
        print(f"  {name2}: {metrics2[key]:.4f}")
        print()


def calculate_rejection_rate(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        if f.get("rejection") is None:
            return 0.0  # No rejection in the dataset
        rejection = f["rejection"][:]
    return rejection.mean()
