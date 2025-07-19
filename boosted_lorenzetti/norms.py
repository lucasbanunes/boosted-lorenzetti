import numpy.typing as npt
import numpy as np


def norm1(data: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    L1 norm

    Parameters
    ----------
    data : npt.NDArray[np.floating]
        Data to normalize

    Returns
    -------
    npt.NDArray[np.floating]
        Normalized data
    """
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data/norms[:, None]
