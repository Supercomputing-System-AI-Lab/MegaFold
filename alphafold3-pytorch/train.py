import argparse
import os

import torch
from loguru import logger

from omnifold.configs import create_trainer_from_conductor_yaml


def main(config_path: str, trainer_name: str):
    """Main function for training.

    :param config_path: Path to a conductor config file.
    :param trainer_name: Name of the trainer to use.
    """
    assert os.path.exists(config_path), f"Config file not found at {config_path}."
    torch.set_float32_matmul_precision("high")

    trainer = create_trainer_from_conductor_yaml(config_path, trainer_name=trainer_name)
    trainer.load_from_checkpoint_folder()

    logger.info("Trainer starting!")
    trainer()
    logger.info("Trainer finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OmniFold.")
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
    args = parser.parse_args()
    main(args.config, args.trainer_name)
