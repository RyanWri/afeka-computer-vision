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
