import inspect
import secrets
import string
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
from beartype.typing import Any, Dict, Iterable, List, Literal
from lightning.fabric.loggers import CSVLogger, Logger, TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger

from omnifold.utils.utils import exists, not_exists


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        """Return the next element from the iterable, cycling through it indefinitely."""
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> "CycleIterator":
        """Return the iterator object itself."""
        return self


def get_default_supported_precision(training: bool) -> str:
    """
    Return the default precision that is supported by the hardware: either `bf16-mixed`, `bf16-true, `16-mixed`, `16-true`, or `32-true`.

    :param training: If True, returns '-mixed' version of the precision; if False, returns '-true' version.
    :return: The default precision that is suitable for the task and is supported by the hardware.
    """
    import torch

    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "bf16-mixed" if training else "bf16-true"
        else:
            raise NotImplementedError(
                "16-bit mixed precision training is currently not supported."
            )

    return "32-true"


def capture_hparams() -> Dict[str, Any]:
    """Capture the local variables ('hyperparameters') from where this function gets called."""
    caller_frame = inspect.currentframe().f_back
    locals_of_caller = caller_frame.f_locals
    hparams = {}
    for name, value in locals_of_caller.items():
        if value is None or isinstance(value, (int, float, str, bool, Path)):
            hparams[name] = value
        elif is_dataclass(value):
            hparams[name] = asdict(value)
        else:
            hparams[name] = str(value)
    return hparams


def parse_dtype(precision: str | None) -> torch.dtype | None:
    """Parse the `precision` argument and return the corresponding data type."""
    if not_exists(precision):
        return None
    elif precision.startswith("bf16"):
        return torch.bfloat16
    elif precision.startswith("16"):
        return torch.float16
    elif precision.startswith("32"):
        return torch.float32
    raise ValueError(f"Precision must contain prefix 'bf16', '16', or '32', got: {precision!r}")


def parse_devices(devices: str | int) -> int:
    """Parse the `devices` argument and return the number of devices to use."""
    if devices in (-1, "auto"):
        return torch.cuda.device_count() or 1
    if isinstance(devices, int) and devices > 0:
        return devices
    raise ValueError(f"Devices must be 'auto' or a positive integer, got: {devices!r}")


def choose_logger(
    logger_name: Literal["csv", "tensorboard", "wandb"],
    out_dir: str | Path = Path("."),
    log_interval: int = 1,
    **kwargs: Any,
):
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    """Choose a logger based on the `logger_name` argument."""
    (Path(out_dir) / "logs").mkdir(parents=True, exist_ok=True)
    if logger_name == "csv":
        csv_kwargs = {
            k: v
            for (k, v) in kwargs.items()
            if k not in {"entity", "group", "id", "project", "resume", "save_dir", "tags"}
        }
        return CSVLogger(
            root_dir=(out_dir / "logs"),
            flush_logs_every_n_steps=log_interval,
            **csv_kwargs,
        )
    if logger_name == "tensorboard":
        return TensorBoardLogger(root_dir=(out_dir / "logs"), **kwargs)
    if logger_name == "wandb":
        if "save_dir" in kwargs:
            Path(kwargs["save_dir"]).mkdir(parents=True, exist_ok=True)
        return WandbLogger(**kwargs)
    raise ValueError(
        f"`logger_name={logger_name}` is not a valid option. Choose from 'csv', 'tensorboard', 'wandb'."
    )


def generate_id(length: int = 8) -> str:
    """Generate a random base-36 string of `length` digits."""
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_logger_experiment_id(loggers: List[Logger] | None) -> str | None:
    """Get the experiment ID from the first logger that has it, if any does."""
    experiment_id = None

    if exists(loggers):
        for logger in loggers:
            if (
                exists(logger)
                and hasattr(logger, "experiment")
                and hasattr(logger.experiment, "id")
            ):
                experiment_id = logger.experiment.id
                break

    return experiment_id
