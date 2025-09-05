from pathlib import Path
from typing import Any, Dict, List, Tuple
from ROOT import TEnv
import torch.nn as nn
import polars as pl

from .threshold import Threshold


def read_legacy_conf_file_as_df(conf_filepath: Path | str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    if isinstance(conf_filepath, Path):
        conf_filepath = str(conf_filepath)

    conf_file = TEnv(conf_filepath)
    metadata = {
        '__name__': conf_file.GetValue('__name__', ''),
        '__version__': conf_file.GetValue('__version__', ''),
        '__operation__': conf_file.GetValue('__operation__', ''),
        '__signature__': conf_file.GetValue('__signature__', ''),
        'Model__size': int(conf_file.GetValue('Model__size', '-1')),
        'Threshold__size': int(conf_file.GetValue('Threshold__size', '-1')),
    }
    if metadata['Model__size'] != metadata['Threshold__size']:
        raise ValueError(
            f'Model__size ({metadata["Model__size"]}) and Threshold__size ({metadata["Threshold__size"]}) must be equal')
    conf_data = {
        'Model__etmin': [float(val) for val in conf_file.GetValue('Model__etmin', '').split('; ')],
        'Model__etmax': [float(val) for val in conf_file.GetValue('Model__etmax', '').split('; ')],
        'Model__etamin': [float(val) for val in conf_file.GetValue('Model__etamin', '').split('; ')],
        'Model__etamax': [float(val) for val in conf_file.GetValue('Model__etamax', '').split('; ')],
        'Model__path': [val for val in conf_file.GetValue('Model__path', '').split('; ')],
        'Threshold__etmin': [float(val) for val in conf_file.GetValue('Threshold__etmin', '').split('; ')],
        'Threshold__etmax': [float(val) for val in conf_file.GetValue('Threshold__etmin', '').split('; ')],
        'Threshold__etamin': [float(val) for val in conf_file.GetValue('Threshold__etmin', '').split('; ')],
        'Threshold__etamax': [float(val) for val in conf_file.GetValue('Threshold__etmin', '').split('; ')],
        'Threshold__slope': [float(val) for val in conf_file.GetValue('Threshold__slope', '').split('; ')],
        'Threshold__offset': [float(val) for val in conf_file.GetValue('Threshold__offset', '').split('; ')],
        'Threshold__MaxAverageMu': [float(val) for val in conf_file.GetValue('Threshold__MaxAverageMu', '').split('; ')]
    }
    df = pl.from_dict(conf_data)
    return df, metadata


class NeuralRingerMember:

    def __init__(self,
                 model: nn.Module,
                 threshold: Threshold):
        self.model = model
        self.threshold = threshold


class NeuralRingerCommittee:

    def __init__(self,
                 models: list,
                 bins: List[List[float]],
                 bins_cols: List[int | str],
                 **kwargs):

        self.models = models
        self.bins = bins
        self.bins_cols = bins_cols
        self.metadata = kwargs

    @classmethod
    def from_legacy_conf_file(cls, conf_filepath: Path | str):

        if not isinstance(conf_filepath, Path):
            conf_filepath = Path(conf_filepath)

        conf_df, metadata = read_legacy_conf_file_as_df(str(conf_filepath))
        filepath_dir = conf_filepath.parent.absolute()
        for row in conf_df.iter_rows(named=True):
            model_path: Path = filepath_dir / row['Model__path']
            if not model_path.exists():
                raise FileNotFoundError(
                    f'Model file {model_path} does not exist')
