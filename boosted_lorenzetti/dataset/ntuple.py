import pandas as pd
import pyarrow as pa
from typing import Dict, Iterable
from pathlib import Path
import ROOT


RINGS_COL = 'cl_rings'

PYARROW_SCHEMA = pa.schema([
    ('EventNumber', pa.int32()),
    ('RunNumber', pa.int32()),
    ('avgmu', pa.float32()),
    ('cl_eta', pa.float32()),
    ('cl_phi', pa.float32()),
    ('cl_e', pa.float32()),
    ('cl_et', pa.float32()),
    ('cl_deta', pa.float32()),
    ('cl_dphi', pa.float32()),
    ('cl_e0', pa.float32()),
    ('cl_e1', pa.float32()),
    ('cl_e2', pa.float32()),
    ('cl_e3', pa.float32()),
    ('cl_ehad1', pa.float32()),
    ('cl_ehad2', pa.float32()),
    ('cl_ehad3', pa.float32()),
    ('cl_etot', pa.float32()),
    ('cl_e233', pa.float32()),
    ('cl_e237', pa.float32()),
    ('cl_e277', pa.float32()),
    ('cl_emaxs1', pa.float32()),
    ('cl_emaxs2', pa.float32()),
    ('cl_e2tsts1', pa.float32()),
    ('cl_reta', pa.float32()),
    ('cl_rphi', pa.float32()),
    ('cl_rhad', pa.float32()),
    ('cl_rhad1', pa.float32()),
    ('cl_eratio', pa.float32()),
    ('cl_f0', pa.float32()),
    ('cl_f1', pa.float32()),
    ('cl_f2', pa.float32()),
    ('cl_f3', pa.float32()),
    ('cl_weta2', pa.float32()),
    ('cl_rings', pa.list_(pa.float32())),
    ('cl_secondR', pa.float32()),
    ('cl_lambdaCenter', pa.float32()),
    ('cl_fracMax', pa.float32()),
    ('cl_lateralMom', pa.float32()),
    ('cl_longitudinalMom', pa.float32()),
    ('el_eta', pa.float32()),
    ('el_et', pa.float32()),
    ('el_phi', pa.float32()),
    ('el_tight', pa.bool8()),
    ('el_medium', pa.bool8()),
    ('el_loose', pa.bool8()),
    ('el_vloose', pa.bool8()),
    ('seed_eta', pa.float32()),
    ('seed_phi', pa.float32()),
    ('mc_pdgid', pa.list_(pa.float32())),
    ('mc_eta', pa.list_(pa.float32())),
    ('mc_phi', pa.list_(pa.float32())),
    ('mc_e', pa.list_(pa.float32())),
    ('mc_et', pa.list_(pa.float32()))
])

FIELDS = list(PYARROW_SCHEMA.names)


def event_as_python(event) -> Dict[str, float]:
    """
    Convert an event object to a dictionary representation.

    Parameters
    ----------
    event : object
        The event object to convert.

    Returns
    -------
    Dict[str, float]
        A dictionary representation of the event.
    """
    return {
        'EventNumber': event.EventNumber,
        'RunNumber': event.RunNumber,
        'avgmu': event.avgmu,
        'cl_eta': event.cl_eta,
        'cl_phi': event.cl_phi,
        'cl_e': event.cl_e,
        'cl_et': event.cl_et,
        'cl_deta': event.cl_deta,
        'cl_dphi': event.cl_dphi,
        'cl_e0': event.cl_e0,
        'cl_e1': event.cl_e1,
        'cl_e2': event.cl_e2,
        'cl_e3': event.cl_e3,
        'cl_ehad1': event.cl_ehad1,
        'cl_ehad2': event.cl_ehad2,
        'cl_ehad3': event.cl_ehad3,
        'cl_etot': event.cl_etot,
        'cl_e233': event.cl_e233,
        'cl_e237': event.cl_e237,
        'cl_e277': event.cl_e277,
        'cl_emaxs1': event.cl_emaxs1,
        'cl_emaxs2': event.cl_emaxs2,
        'cl_e2tsts1': event.cl_e2tsts1,
        'cl_reta': event.cl_reta,
        'cl_rphi': event.cl_rphi,
        'cl_rhad': event.cl_rhad,
        'cl_rhad1': event.cl_rhad1,
        'cl_eratio': event.cl_eratio,
        'cl_f0': event.cl_f0,
        'cl_f1': event.cl_f1,
        'cl_f2': event.cl_f2,
        'cl_f3': event.cl_f3,
        'cl_weta2': event.cl_weta2,
        'cl_rings': list(event.cl_rings),
        'cl_secondR': event.cl_secondR,
        'cl_lambdaCenter': event.cl_lambdaCenter,
        'cl_fracMax': event.cl_fracMax,
        'cl_lateralMom': event.cl_lateralMom,
        'cl_longitudinalMom': event.cl_longitudinalMom,
        'el_eta': event.el_eta,
        'el_et': event.el_et,
        'el_phi': bool(event.el_phi),
        'el_tight': bool(event.el_tight),
        'el_medium': bool(event.el_medium),
        'el_loose': bool(event.el_loose),
        'el_vloose': bool(event.el_vloose),
        'seed_eta': event.seed_eta,
        'seed_phi': event.seed_phi,
        'mc_pdgid': list(event.mc_pdgid),
        'mc_eta': list(event.mc_eta),
        'mc_phi': list(event.mc_phi),
        'mc_e': list(event.mc_e),
        'mc_et': list(event.mc_et),
    }


def to_pdf(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
           ttree_name: str = 'physics') -> pd.DataFrame:
    """
    Convert a single ntuple root file to a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path
        The path to the input ntuple file.
    output_file : Path | str
        The path where the output DataFrame will be saved.
    """
    if not isinstance(input_file, ROOT.TChain):
        if isinstance(input_file, Path):
            input_file = [str(input_file)]
        elif isinstance(input_file, str):
            input_file = [input_file]
        chain = ROOT.TChain(ttree_name)
        for file in input_file:
            if isinstance(file, Path):
                file = str(file)
            chain.Add(file)
    data = {col_name: [] for col_name in FIELDS}
    for event in chain:
        event_data = event_as_python(event)
        for col_name, value in event_data.items():
            data[col_name].append(value)
    return pd.DataFrame(data)


def to_parquet(input_file: str | Path,
               output_file: Path | str,
               ttree_name: str = 'physics') -> None:
    """
    Convert a single ntuple root file to a ntuple parquet file.

    Parameters
    ----------
    input_file : str | Path
        The path to the input ntuple file.
    output_file : Path | str
        The path where the output parquet file will be saved.
    """
    to_pdf(input_file, ttree_name).to_parquet(output_file, index=False, compression='gzip')
