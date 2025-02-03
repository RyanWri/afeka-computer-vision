from src.experiment import run_experiment
from src.inference import run_inference
from src.io_utils import dict_to_dataframe, load_config


if __name__ == "__main__":
    config_path = "rejection_gate.yaml"
    config = load_config(config_path)

    # run inference for baseline
    if config["baseline_model"]["enabled"]:
        results = run_inference(config, threshold=0.5)
        dict_to_dataframe(
            results, config["baseline_model"]["inference"]["original_results"]
        )

    # Run Experiment if enabled
    run_experiment(config)
