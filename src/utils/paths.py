import os


def get_config_path(relative_path):
    """
    Construct the absolute path for a configuration file.

    Args:
        relative_path (str): Path to the config file relative to the 'config' directory.

    Returns:
        str: Absolute path to the configuration file.
    """
    src_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of this script
    config_dir = os.path.join(
        src_dir, "../config"
    )  # Relative path to the config directory
    return os.path.normpath(os.path.join(config_dir, relative_path))


def resolve_experiment_paths(config):
    """
    Dynamically resolve all paths for results and metrics based on the base root and experiment alias.

    Args:
        config (dict): Experiment configuration.

    Returns:
        dict: Updated configuration with resolved paths for results and metrics.
    """
    base_root = config["experiment"]["base_root"]
    alias = config["experiment"]["alias"]

    # Create experiment-specific directory under base_root
    experiment_root = os.path.join(base_root, alias)
    os.makedirs(experiment_root, exist_ok=True)

    # Update paths for outputs
    resolved_paths = {
        "results_path": os.path.join(experiment_root, "results.parquet"),
        "metrics_path": os.path.join(experiment_root, "metrics.json"),
    }

    # Inject resolved paths into the configuration
    config["experiment"]["results_path"] = resolved_paths["results_path"]
    config["experiment"]["metrics_path"] = resolved_paths["metrics_path"]

    return config
