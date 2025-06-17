import argparse
import os

import rootutils
from beartype.typing import Literal

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from megafold.configs import create_trainer_from_conductor_yaml


def cache_mmcif_msas(config_path: str, trainer_name: str, split: Literal["train", "val", "test"]):
    """Main function for mmCIF MSA caching.

    :param config_path: Path to a conductor config file.
    :param trainer_name: Name of the trainer to use.
    """
    """Cache MSA of each mmCIF file in a cache directory."""
    assert os.path.exists(config_path), f"Config file not found at {config_path}."

    trainer = create_trainer_from_conductor_yaml(config_path, trainer_name=trainer_name)
    trainer.cache_msas(split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache mmCIF MSAs.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a conductor config file.",
    )
    parser.add_argument(
        "--trainer_name",
        type=str,
        required=True,
        help="Name of the trainer to use.",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Split for which to cache MSAs.",
    )
    args = parser.parse_args()
    cache_mmcif_msas(args.config, args.trainer_name, args.split)
