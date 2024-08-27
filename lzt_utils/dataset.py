import os
import ROOT
import glob
import json
from hep_utils.root import rdf_to_pandas
import pandas as pd

FILE_DIRECTORIES = [
    'EVT',
    'HIT',
    'ESD',
    'AOD',
    'NTUPLE'
]


class LztDataset:
    def __init__(self,
                 path: str,
                 basename: str,
                 label: str = None):
        self.path = path
        self.basename = basename
        self.label = label

    @property
    def evt_path(self) -> str:
        return os.path.join(self.path, 'EVT')

    @property
    def hit_path(self) -> str:
        return os.path.join(self.path, 'HIT')

    @property
    def esd_path(self) -> str:
        return os.path.join(self.path, 'ESD')

    @property
    def aod_path(self) -> str:
        return os.path.join(self.path, 'AOD')

    @property
    def ntuple_path(self) -> str:
        return os.path.join(self.path, 'NTUPLE')

    def get_ntuple_rdf(self) -> ROOT.RDataFrame:
        ntuple_files = glob.glob(self.ntuple_path + '/*.root')
        rdf = ROOT.RDataFrame("events", ntuple_files)
        return rdf

    def get_ntuple_pdf(self) -> pd.DataFrame:
        return rdf_to_pandas(self.get_ntuple_rdf())

    def create_dir(self, directory: str) -> str:
        abspath = os.path.join(self.path, directory)
        os.makedirs(abspath, exist_ok=True)
        return abspath

    @classmethod
    def from_dir(cls, path: str):
        if os.path.isdir(path):
            with open(os.path.join(path, 'dataset_info.json'), 'r') as f:
                data = json.load(f)
                return cls(path, **data)
        else:
            raise ValueError(f'{path} is not a directory')
