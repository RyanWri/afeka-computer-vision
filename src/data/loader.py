# Data loading and augmentation logic
import logging
from pydantic import BaseModel
from datasets import load_dataset

logging.basicConfig(level=logging.DEBUG)


class Split(BaseModel):
    Train = "train"
    Test: str = "test"
    Validation: str = "validation"


def load_data(pcam_source, split):
    dataset = load_dataset(pcam_source, split=split)


if __name__ == "__main__":
    pcam_source = "dpdl-benchmark/patch_camelyon"
    split = Split.Validation
    load_data(pcam_source, split)
