from numbers import Number
import numpy as np


def deltaR(eta1: Number, phi1: Number, eta2: Number, phi2: Number) -> float:
    """
    Computes the deltaR between two objects in eta-phi space.

    Parameters
    ----------
    eta1 : Number
        eta coordinate of the first object
    phi1 : Number
        phi coordinate of the first object
    eta2 : Number
        eta coordinate of the second object
    phi2 : Number
        phi coordinate of the second object

    Returns
    -------
    float
        deltaR between the two objects
    """
    deta = np.abs(eta1 - eta2)
    dphi = np.abs(phi1 - phi2)
    if isinstance(eta1, np.ndarray):
        dphi[dphi >= np.pi] = 2*np.pi - dphi[dphi >= np.pi]
    elif dphi >= np.pi:
        dphi = 2*np.pi - dphi
    return np.sqrt((deta**2) + (dphi**2))
