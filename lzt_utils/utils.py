from typing import Literal, Union, Iterable, Iterator
import os
from glob import iglob
from numbers import Number
import logging


def open_directories(
        paths: Union[Iterable[str], str],
        file_ext: str,
        dev: bool = False) -> Iterator[str]:
    """
    Generator that opens all directories in an iterator for
    a specific file extension. This is useful for script cases where
    an user can pass a mix of directories and filepaths.

    Parameters
    ----------
    paths : Union[Iterable[str], str]
        Path or paths to look for files with file_ext
    file_ext : str
        Te desired file extension to look for
    dev: bool
        If True, the function will yield just the first file found

    Yields
    ------
    str
        The path to a file

    Raises
    ------
    ValueError
        Raised if there is a file that does not have file_ext as its extension
    """
    if isinstance(paths, str):
        paths = [paths]
    for i, ipath in enumerate(paths):
        if os.path.isdir(ipath):
            dir_paths = iglob(
                os.path.join(ipath, '**', f'*.{file_ext}'),
                recursive=True
            )
            for j, open_path in enumerate(dir_paths):
                if dev and j > 0:
                    break
                else:
                    yield open_path
        elif ipath.endswith(f'.{file_ext}'):
            yield ipath
        else:
            raise ValueError(
                f'{ipath} does not have the expected {file_ext} extension'
            )
        if dev and i > 0:
            break


def is_between(value: Number, low: Number, high: Number,
               inclusive: Literal['both', 'neither', 'left', 'right']
               ) -> bool:
    """
    Check if a value is between two values.

    Parameters
    ----------
    value : Number
        The value to check.
    low : Number
        The lower bound.
    high : Number
        The upper bound.

    Returns
    -------
    bool
        True if the value is between low and high, False otherwise.
    """
    if inclusive == 'both':
        return (low <= value) & (value <= high)
    elif inclusive == 'neither':
        return (low < value) & (value < high)
    elif inclusive == 'left':
        return (low <= value) & (value < high)
    elif inclusive == 'right':
        return (low < value) & (value <= high)
    else:
        raise ValueError(f'{inclusive} is not a valid inclusive option')


def set_logger(level="INFO"):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(module)s"
                " | %(lineno)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "default",
                "stream": "ext://sys.stdout"
            },
        },
        "loggers": {
            "": {
                "level": level,
                "handlers": ["stdout"]
            }
        }
    }
    logging.config.dictConfig(logging_config)