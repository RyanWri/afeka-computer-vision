import os
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def concat_and_process_results(base_folder, original_file, experiment_file, confidence):
    """
    Loads and processes result data from CSV files in the given base folder.

    :param base_folder: Path to the folder containing result files
    :param confidence: Threshold for rejection
    :return: Processed DataFrame
    """
    original_result = os.path.join(base_folder, original_file)
    df = pd.read_csv(original_result, index_col=0)

    equal_weights = os.path.join(base_folder, experiment_file)
    df_equal = pd.read_csv(equal_weights, index_col=0)

    # Concatenate dataframes
    data = pd.concat([df, df_equal], axis=1)

    # Handle dtypes
    data["true_label"] = data["true_label"].astype(int)
    data["reject_score"] = data[["knn", "margin", "mahalanobis"]].sum(axis=1)
    data["rejected"] = data["reject_score"] > confidence

    return data


def calculate_metrics(df: pd.DataFrame, label_col: str, pred_col: str) -> dict:
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
