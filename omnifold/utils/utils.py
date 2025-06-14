import shutil
import subprocess  # nosec
from functools import wraps
from pathlib import Path

import numpy as np
from beartype.typing import Any, Callable, Iterable, OrderedDict


def exists(val: Any) -> bool:
    """Check if a value exists.

    :param val: The value to check.
    :return: `True` if the value exists, otherwise `False`.
    """
    return val is not None


def not_exists(val: Any) -> bool:
    """Check if a value does not exist.

    :param val: The value to check.
    :return: `True` if the value does not exist, otherwise `False`.
    """
    return val is None


def default(v: Any, d: Any) -> Any:
    """Return default value `d` if `v` does not exist (i.e., is `None`).

    :param v: The value to check.
    :param d: The default value to return if `v` does not exist.
    :return: The value `v` if it exists, otherwise the default value `d`.
    """
    return v if exists(v) else d


def first(arr: Iterable[Any]) -> Any:
    """Return the first element of an iterable object such as a list.

    :param arr: An iterable object.
    :return: The first element of the iterable object.
    """
    return arr[0]


def always(value):
    """Always return a value."""

    def inner(*args, **kwargs):
        """Inner function."""
        return value

    return inner


def identity(x, *args, **kwargs):
    """Return the input value."""
    return x


def np_mode(x: np.ndarray) -> Any:
    """Return the mode of a 1D NumPy array."""
    assert x.ndim == 1, f"Input NumPy array must be 1D, not {x.ndim}D."
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def maybe_cache(
    fn: Callable, *, cache: dict, key: str | None, should_cache: bool = True
) -> Callable:
    """Cache a function's output based on a key and a cache dictionary."""
    if not should_cache or not exists(key):
        return fn

    @wraps(fn)
    def inner(*args, **kwargs):
        if key in cache:
            return cache[key]

        out = fn(*args, **kwargs)

        cache[key] = out
        return out

    return inner


def is_float(x: Any) -> bool:
    """Check if a value is a float."""
    try:
        float(x)
        return True
    except ValueError:
        return False


def is_int(x: Any) -> bool:
    """Check if a value is an integer."""
    try:
        int(x)
        return True
    except ValueError:
        return False


def remove_directory(dir_path: str | Path):
    """Remove a directory and all its contents if it exists."""
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)

    if dir_path.exists():
        shutil.rmtree(dir_path)


def exec_command(cmd: str, stdout=subprocess.PIPE, stderr=subprocess.PIPE) -> str:
    """Execute a shell command and return `stdout` if it finishes with exit code `0`.

    :param cmd: Shell command to be executed
    :param stdout, stderr: Streams to capture process output.
        Default value (PIPE) intercepts process output, setting to None
        blocks this.
    :return: `stdout` if the process finishes with exit code `0`.
    """

    proc = subprocess.Popen(cmd, shell=True, stdout=stdout, stderr=stderr)  # nosec
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"FAILED: {cmd}\n{err}")
    return out.decode("utf8") if out is not None else None


def apply_function_to_ordered_dict_keys(ordered_dict: OrderedDict, func: Callable) -> OrderedDict:
    """Apply a function to the keys of an ordered dictionary in-place.

    :param ordered_dict: An ordered dictionary.
    :param func: A function to apply to the keys.
    :return: The ordered dictionary with the function applied to the keys.
    """
    keys = list(ordered_dict.keys())
    for key in keys:
        ordered_dict[func(key)] = ordered_dict.pop(key)
    return ordered_dict
