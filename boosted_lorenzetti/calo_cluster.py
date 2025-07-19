from typing import TypedDict, Tuple


class CaloCluster(TypedDict):
    e: float
    et: float
    eta: float
    phi: float
    deta: float
    dphi: float
    e0: float
    e1: float
    e2: float
    e3: float
    ehad1: float
    ehad2: float
    ehad3: float
    etot: float
    e233: float
    e237: float
    e277: float
    emaxs1: float
    emaxs2: float
    e2tsts1: float
    reta: float
    rphi: float
    rhad: float
    rhad1: float
    eratio: float
    f0: float
    f1: float
    f2: float
    f3: float
    weta2: float
    secondR: float
    lambdaCenter: float
    secondLambda: float
    fracMax: float
    lateralMom: float
    longitudinalMom: float


def get_eta_phi_range(cluster: CaloCluster) -> Tuple[Tuple[float, float],
                                                     Tuple[float, float]]:
    """
    Get the eta and phi range of a cluster.

    Parameters
    ----------
    cluster : CaloCluster
        The cluster.

    Returns
    -------
    Tuple[Tuple[float, float], Tuple[float, float]]
        The eta and phi range of the cluster.
    """
    eta_range = (
        cluster.eta - cluster.deta,
        cluster.eta + cluster.deta
    )
    phi_range = (
        cluster.phi - cluster.dphi,
        cluster.phi + cluster.dphi
    )
    return eta_range, phi_range
