import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_rejection_metrics(results_df):
    """
    Compute metrics for the rejection mechanism.

    Args:
        results_df (pd.DataFrame): DataFrame containing experiment results.

    Returns:
        dict: Flattened metrics for the rejection mechanism.
    """
    # True labels for rejection (1 if rejected, 0 if not)
    true_rejection_labels = results_df["reject"].astype(int)
    return {"rejection_rate": true_rejection_labels.mean()}


def compute_non_rejected_metrics(results_df):
    """
    Compute classification metrics for non-rejected samples.

    Args:
        results_df (pd.DataFrame): DataFrame containing experiment results.

    Returns:
        dict: Flattened metrics for classification of non-rejected samples.
    """
    # Filter non-rejected samples
    non_rejected = results_df[not results_df["reject"]]

    # Extract labels and predictions
    true_labels = non_rejected["label"].values
    probabilities = (
        non_rejected["probability"].dropna().values
    )  # Predicted probabilities
    predictions = non_rejected["predicted_label"].dropna().values  # Binary predictions

    # Classification metrics
    accuracy = (predictions == true_labels).mean()
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc_classification = roc_auc_score(true_labels, probabilities)

    return {
        "classification_accuracy": accuracy,
        "classification_precision": precision,
        "classification_recall": recall,
        "classification_f1_score": f1,
        "classification_auc": auc_classification,
    }


def summarize_metrics(results_df):
    """
    Compute and summarize metrics for rejection and classification.

    Args:
        results_df (pd.DataFrame): DataFrame containing experiment results.

    Returns:
        dict: Flattened dictionary containing all metrics.
    """
    rejection_metrics = compute_rejection_metrics(results_df)
    classification_metrics = compute_non_rejected_metrics(results_df)

    # Combine and flatten both metric dictionaries
    return {**rejection_metrics, **classification_metrics}


def calculate_metrics(file_path: str, label_col: str, pred_col: str) -> dict:
    df = df = pd.read_csv(file_path, index_col=0)
    df[label_col] = df[label_col].astype(int)
    df[pred_col] = df[pred_col].astype(int)

    y_true = df[label_col].values
    y_pred = df[pred_col].values

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="binary"
        ),  # Adjust average for multi-class
        "recall": recall_score(y_true, y_pred, average="binary"),
        "f1_score": f1_score(y_true, y_pred, average="binary"),
    }
    return metrics
