# Data loading and augmentation logic
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.DEBUG)


class Split:
    Train = "train"
    Test = "test"
    Validation = "validation"


def load_data(pcam_source, split):
    dataset = load_dataset(pcam_source, split=split)
    return dataset


if __name__ == "__main__":
    pcam_source = "dpdl-benchmark/patch_camelyon"
    split = Split.Validation
    data = load_data(pcam_source, split)
    print(data["label"])
