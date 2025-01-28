import pandas as pd
from src.models.rejection_gate import RejectionGate
from src.models.train_models import get_features


def run_experiment(config):
    """
    Run the experiment pipeline based on the loaded configuration.
    """
    split = config["input"]["split"]
    features = get_features(config, split)

    # Initialize rejection gate
    rejection_gate = RejectionGate(config["rejection_models"])

    # Prepare results storage
    results = []

    for feature in features:
        reject_score = rejection_gate.compute_rejection_confidence(feature)
        results.append(reject_score)

    # Save results to Parquet
    baseline_model_config = config["baseline_model"]
    baseline_results_df = pd.read_parquet(baseline_model_config["original_results"])
    baseline_results_df["probability"] = round(baseline_results_df["probability"], 5)
    baseline_results_df["margin"] = abs(baseline_results_df["probability"] - 0.5)
    baseline_results_df["margin"] = baseline_results_df["margin"].astype(float)

    results_path = config["experiment"]["results_path"]
    results_df = baseline_results_df
    results_df["reject_score"] = results
    results_df.to_parquet(results_path, engine="pyarrow")
    print(f"Experiment results saved to {results_path}")
