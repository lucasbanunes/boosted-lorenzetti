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
