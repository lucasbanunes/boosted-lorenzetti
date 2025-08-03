from typing import Literal, Iterable, Iterator, Generator, Any
from numbers import Number
import logging
from pathlib import Path


def open_directories(
        paths: str | Path | Iterable[str | Path],
        file_ext: str,
        dev: bool = False) -> Iterator[Path]:
    """
    Generator that opens all directories in an iterator for
    a specific file extension. This is useful for script cases where
    an user can pass a mix of directories and filepaths.

    Parameters
    ----------
    paths : str | Path | Iterable[str | Path]
        A single path or an iterable of paths. These can be directories or
        file paths. If a directory is provided, it will search recursively
        for files with the specified file extension.
    file_ext : str
        The desired file extension to look for
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
        paths = [Path(paths)]
    elif isinstance(paths, Path):
        paths = [paths]
    for i, ipath in enumerate(paths):
        if isinstance(ipath, str):
            ipath = Path(ipath)
        if dev and i > 0:
            break
        elif ipath.is_dir():
            dir_paths = ipath.glob(f'**/*.{file_ext}')
            if not dir_paths:
                continue
            for j, open_path in enumerate(dir_paths):
                if dev and j > 0:
                    break
                else:
                    yield open_path
        elif ipath.suffix == f'.{file_ext}':
            yield ipath
        else:
            raise ValueError(
                f'{ipath} does not have the expected {file_ext} extension'
            )


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


def set_logger(level="INFO", name=""):
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
            name: {
                "level": level,
                "handlers": ["stdout"]
            }
        }
    }
    logging.config.dictConfig(logging_config)


def iterable_to_generator(iterable: Iterable[Any]) -> Generator[Any, None, None]:
    """
    Converts an iterable to a generator.

    Parameters
    ----------
    iterable : Iterable[Any]
        The iterable to be converted into a generator.

    Returns
    -------
    Generator[Any, None, None]
        A generator that yields items from the input iterable.
    """
    yield from iterable


def list_by_pattern(files: Iterable[Path | str], pattern: str) -> Generator[Path, None, None]:
    """
    Get files from an iterable of paths that match a specific pattern.

    Parameters
    ----------
    files : Iterable[Path | str]
        An iterable of Path objects or strings to search for matching files.
    pattern : str
        The pattern to match (e.g., '*.parquet').

    Returns
    -------
    Generator[Path, None, None]
        A generator that yields paths of matching files.
    """
    for file in files:
        if not isinstance(file, Path):
            file = Path(file)
        if file.is_dir():
            yield from file.glob(pattern)
        else:
            if file.match(pattern):
                yield file


def unflatted_dict_process_value(value: Any) -> Any:
    """
    Process a value for unflattened dictionary.

    Parameters
    ----------
    value : Any
        The value to process.

    Returns
    -------
    Any
        The processed value.
    """
    if isinstance(value, dict):
        return unflatten_dict(value)
    elif isinstance(value, (list, tuple)):
        return [unflatted_dict_process_value(item) for item in value]
    elif isinstance(value, set):
        return {unflatted_dict_process_value(item) for item in value}
    else:
        return value


def unflatten_dict(flat_dict: dict[str, Any], separator: str = '.') -> dict[str, Any]:
    """
    Unflatten a dictionary with keys that are concatenated by a separator.

    Parameters
    ----------
    flat_dict : dict[str, Any]
        The flat dictionary to unflatten.
    separator : str, optional
        The separator used in the keys of the flat dictionary. Default is '.'.

    Returns
    -------
    dict[str, Any]
        The unflattened dictionary.
    """
    unflattened = {}
    for key, value in flat_dict.items():

        if '.' not in key:
            unflattened[key] = unflatted_dict_process_value(value)
            continue

        # Split the key by the separator and build the nested structure
        parts = key.split(separator)
        d = unflattened
        for part in parts[:-1]:
            # This works because d is passed by reference
            d = d.setdefault(part, {})
        d[parts[-1]] = unflatted_dict_process_value(value)

    return unflattened
