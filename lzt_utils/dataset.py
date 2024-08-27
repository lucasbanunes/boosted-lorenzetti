from dataclasses import dataclass
import os

FILE_DIRECTORIES = [
    'EVT',
    'HIT',
    'ESD',
    'AOD'
]


@dataclass
class LztDataset:
    path: str

    @property
    def evt_path(self):
        return os.path.join(self.path, 'EVT')

    @property
    def hit_path(self):
        return os.path.join(self.path, 'HIT')

    @property
    def esd_path(self):
        return os.path.join(self.path, 'ESD')

    @property
    def aod_path(self):
        return os.path.join(self.path, 'AOD')
