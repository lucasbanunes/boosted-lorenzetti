import os
import ROOT
import json
from lzt_utils.root import rdf_to_pandas, rdf_column_names
import pandas as pd
from pathlib import Path
from typing import Iterator, Union, Dict, List, Any
import awkward as ak
import numpy as np

FILE_DIRECTORIES = [
    'EVT',
    'HIT',
    'ESD',
    'AOD',
    'NTUPLE'
]

ESD_STRUCTS = [
    'CaloCellContainer_Cells',
    'CaloDetDescriptorContainer_Cells',
    'EventInfoContainer_Events',
    'EventSeedContainer_Seeds',
    'TruthParticleContainer_Particles'
]

ESD_STRUCT_FILEDS = [
    'CaloCellContainer_Cells.descriptor_link',
    'CaloCellContainer_Cells.deta',
    'CaloCellContainer_Cells.dphi',
    'CaloCellContainer_Cells.e',
    'CaloCellContainer_Cells.et',
    'CaloCellContainer_Cells.eta',
    'CaloCellContainer_Cells.phi',
    'CaloCellContainer_Cells.tau',
    'CaloDetDescriptorContainer_Cells.bc_duration',
    'CaloDetDescriptorContainer_Cells.bcid_end',
    'CaloDetDescriptorContainer_Cells.bcid_start',
    'CaloDetDescriptorContainer_Cells.cell_link',
    'CaloDetDescriptorContainer_Cells.deta',
    'CaloDetDescriptorContainer_Cells.detector',
    'CaloDetDescriptorContainer_Cells.dphi',
    'CaloDetDescriptorContainer_Cells.e',
    'CaloDetDescriptorContainer_Cells.edep',
    'CaloDetDescriptorContainer_Cells.edep_per_bunch',
    'CaloDetDescriptorContainer_Cells.eta',
    'CaloDetDescriptorContainer_Cells.hash',
    'CaloDetDescriptorContainer_Cells.phi',
    'CaloDetDescriptorContainer_Cells.pulse',
    'CaloDetDescriptorContainer_Cells.sampling',
    'CaloDetDescriptorContainer_Cells.tau',
    'CaloDetDescriptorContainer_Cells.tof',
    'CaloDetDescriptorContainer_Cells.z',
    'EventInfoContainer_Events.avgmu',
    'EventInfoContainer_Events.eventNumber',
    'EventInfoContainer_Events.runNumber',
    'EventSeedContainer_Seeds.e',
    'EventSeedContainer_Seeds.et',
    'EventSeedContainer_Seeds.eta',
    'EventSeedContainer_Seeds.id',
    'EventSeedContainer_Seeds.phi',
    'TruthParticleContainer_Particles.e',
    'TruthParticleContainer_Particles.et',
    'TruthParticleContainer_Particles.eta',
    'TruthParticleContainer_Particles.pdgid',
    'TruthParticleContainer_Particles.phi',
    'TruthParticleContainer_Particles.px',
    'TruthParticleContainer_Particles.py',
    'TruthParticleContainer_Particles.pz',
    'TruthParticleContainer_Particles.seedid',
    'TruthParticleContainer_Particles.vx',
    'TruthParticleContainer_Particles.vy',
    'TruthParticleContainer_Particles.vz'
]


AOD_STRUCTS = [
    'CaloCellContainer_Cells',
    'CaloClusterContainer_Clusters',
    'CaloDetDescriptorContainer_Cells',
    'CaloRingsContainer_Rings',
    'ElectronContainer_Electrons',
    'EventInfoContainer_Events',
    'EventSeedContainer_Seeds',
    'TruthParticleContainer_Particles'
]


AOD_STRUCT_FIELDS = [
    'CaloCellContainer_Cells.descriptor_link',
    'CaloCellContainer_Cells.deta',
    'CaloCellContainer_Cells.dphi',
    'CaloCellContainer_Cells.e',
    'CaloCellContainer_Cells.et',
    'CaloCellContainer_Cells.eta',
    'CaloCellContainer_Cells.phi',
    'CaloCellContainer_Cells.tau',
    'CaloClusterContainer_Clusters.cell_links',
    'CaloClusterContainer_Clusters.deta',
    'CaloClusterContainer_Clusters.dphi',
    'CaloClusterContainer_Clusters.e',
    'CaloClusterContainer_Clusters.e0',
    'CaloClusterContainer_Clusters.e1',
    'CaloClusterContainer_Clusters.e2',
    'CaloClusterContainer_Clusters.e233',
    'CaloClusterContainer_Clusters.e237',
    'CaloClusterContainer_Clusters.e277',
    'CaloClusterContainer_Clusters.e2tsts1',
    'CaloClusterContainer_Clusters.e3',
    'CaloClusterContainer_Clusters.ehad1',
    'CaloClusterContainer_Clusters.ehad2',
    'CaloClusterContainer_Clusters.ehad3',
    'CaloClusterContainer_Clusters.emaxs1',
    'CaloClusterContainer_Clusters.emaxs2',
    'CaloClusterContainer_Clusters.eratio',
    'CaloClusterContainer_Clusters.et',
    'CaloClusterContainer_Clusters.eta',
    'CaloClusterContainer_Clusters.etot',
    'CaloClusterContainer_Clusters.f0',
    'CaloClusterContainer_Clusters.f1',
    'CaloClusterContainer_Clusters.f2',
    'CaloClusterContainer_Clusters.f3',
    'CaloClusterContainer_Clusters.fracMax',
    'CaloClusterContainer_Clusters.lambdaCenter',
    'CaloClusterContainer_Clusters.lateralMom',
    'CaloClusterContainer_Clusters.longitudinalMom',
    'CaloClusterContainer_Clusters.phi',
    'CaloClusterContainer_Clusters.reta',
    'CaloClusterContainer_Clusters.rhad',
    'CaloClusterContainer_Clusters.rhad1',
    'CaloClusterContainer_Clusters.rphi',
    'CaloClusterContainer_Clusters.secondLambda',
    'CaloClusterContainer_Clusters.secondR',
    'CaloClusterContainer_Clusters.weta2',
    'CaloDetDescriptorContainer_Cells.bc_duration',
    'CaloDetDescriptorContainer_Cells.bcid_end',
    'CaloDetDescriptorContainer_Cells.bcid_start',
    'CaloDetDescriptorContainer_Cells.cell_link',
    'CaloDetDescriptorContainer_Cells.deta',
    'CaloDetDescriptorContainer_Cells.detector',
    'CaloDetDescriptorContainer_Cells.dphi',
    'CaloDetDescriptorContainer_Cells.e',
    'CaloDetDescriptorContainer_Cells.edep',
    'CaloDetDescriptorContainer_Cells.edep_per_bunch',
    'CaloDetDescriptorContainer_Cells.eta',
    'CaloDetDescriptorContainer_Cells.hash',
    'CaloDetDescriptorContainer_Cells.phi',
    'CaloDetDescriptorContainer_Cells.pulse',
    'CaloDetDescriptorContainer_Cells.sampling',
    'CaloDetDescriptorContainer_Cells.tau',
    'CaloDetDescriptorContainer_Cells.tof',
    'CaloDetDescriptorContainer_Cells.z',
    'CaloRingsContainer_Rings.cluster_link',
    'CaloRingsContainer_Rings.rings',
    'ElectronContainer_Electrons.cluster_link',
    'ElectronContainer_Electrons.decisions',
    'ElectronContainer_Electrons.e',
    'ElectronContainer_Electrons.et',
    'ElectronContainer_Electrons.eta',
    'ElectronContainer_Electrons.phi',
    'EventInfoContainer_Events.avgmu',
    'EventInfoContainer_Events.eventNumber',
    'EventInfoContainer_Events.runNumber',
    'EventSeedContainer_Seeds.e',
    'EventSeedContainer_Seeds.et',
    'EventSeedContainer_Seeds.eta',
    'EventSeedContainer_Seeds.id',
    'EventSeedContainer_Seeds.phi',
    'TruthParticleContainer_Particles.e',
    'TruthParticleContainer_Particles.et',
    'TruthParticleContainer_Particles.eta',
    'TruthParticleContainer_Particles.pdgid',
    'TruthParticleContainer_Particles.phi',
    'TruthParticleContainer_Particles.px',
    'TruthParticleContainer_Particles.py',
    'TruthParticleContainer_Particles.pz',
    'TruthParticleContainer_Particles.seedid',
    'TruthParticleContainer_Particles.vx',
    'TruthParticleContainer_Particles.vy',
    'TruthParticleContainer_Particles.vz'
]


def esd_rdf_to_ak(rdf: ROOT.RDataFrame,
                  columns: List[str] = None,
                  ak_kwargs: Dict[str, Any] = {}
                  ) -> ak.Array:
    if columns is None:
        columns = ESD_STRUCT_FILEDS
    rdf_ak = ak.from_rdataframe(rdf, columns=columns,
                                **ak_kwargs)
    rdf_ak_dict = {
        key: {} for key in ESD_STRUCTS
    }
    to_int = [
        'EventInfoContainer_Events.eventNumber'
    ]

    for field in rdf_ak.fields:
        # The events from EventInfoContainer are always a list
        # with one element, so we extract the first element.
        main_field, subfield = field.split('.')
        if main_field == 'EventInfoContainer_Events':
            rdf_ak_dict[main_field][subfield] = ak.firsts(rdf_ak[field])
        else:
            rdf_ak_dict[main_field][subfield] = rdf_ak[field]

        if field in to_int:
            rdf_ak_dict[main_field][subfield] = ak.values_astype(
                rdf_ak_dict[main_field][subfield], np.int32)

    rdf_ak = ak.Array({key: ak.zip(val)
                       for key, val in rdf_ak_dict.items()
                       if val})
    return rdf_ak


def aod_rdf_to_ak(rdf: ROOT.RDataFrame,
                  columns: List[str] = None,
                  ak_kwargs: Dict[str, Any] = {}
                  ) -> ak.Array:
    if columns is None:
        columns = rdf_column_names(rdf)
    rdf_ak = ak.from_rdataframe(rdf, columns=columns,
                                **ak_kwargs)
    rdf_ak_dict = {
        key: {} for key in AOD_STRUCTS
    }
    to_int = [
        'EventInfoContainer_Events.eventNumber'
    ]

    for field in rdf_ak.fields:
        # The events from EventInfoContainer are always a list
        # with one element, so we extract the first element.
        main_field, subfield = field.split('.')
        if main_field == 'EventInfoContainer_Events':
            rdf_ak_dict[main_field][subfield] = ak.firsts(rdf_ak[field])
        else:
            rdf_ak_dict[main_field][subfield] = rdf_ak[field]

        if field in to_int:
            rdf_ak_dict[main_field][subfield] = ak.values_astype(
                rdf_ak_dict[main_field][subfield], np.int32)

    rdf_ak = ak.Array({key: ak.zip(val)
                       for key, val in rdf_ak_dict.items()
                       if val})
    return rdf_ak


class LztDataset:
    """
    Class for managing the dataset directories generated by lorenzetti

    Attributes
    ----------
    path : str
        Path to the dataset directory
    basename : str
        Basename of the dataset files
    label : str, optional
        Dataset label, useful for plotting, by default None

    Properties
    ----------
    evt_path : str
        Path to the EVT directory
    hit_path : str
        Path to the HIT directory
    esd_path : str
        Path to the ESD directory
    aod_path : str
        Path to the AOD directory
    ntuple_path : str
        Path to the NTUPLE directory
    """

    def __init__(self,
                 path: Union[str, Path],
                 basename: str,
                 label: str = None,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            Path to the dataset directory
        basename : str
            Basename of the dataset files
        label : str, optional
            Dataset label, useful for plotting, by default None
        """
        if isinstance(path, str):
            self.path = Path(path)
        self.path = path
        self.basename = basename
        self.label = label
        self.evt_path = self.path / 'EVT'
        self.hit_path = self.path / 'HIT'
        self.esd_path = self.path / 'ESD'
        self.aod_path = self.path / 'AOD'
        self.ntuple_path = self.path / 'NTUPLE'
        self.__hit_event_counter = None
        self.__esd_event_counter = None

    def __repr__(self) -> str:
        repr_str = f'LztDataset(path={self.path}, label={self.label})'
        return repr_str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def hit_files(self) -> Iterator[Path]:
        """
        Iterator over the HIT files

        Returns
        -------
        Iterator[Path]
            Iterator over the HIT files
        """
        return self.hit_path.glob('*.root')

    @property
    def hit_event_counter(self) -> Dict[str, int]:
        """
        Number of completed HIT events

        Returns
        -------
        Dict[str, int]
            Dictionary with the number of events
            from hit files. Has the following schema:
            {
                'Event': int,
                'Completed': int,
                'Timeout': int
            }
        """
        if self.__hit_event_counter is None:
            self.__hit_event_counter = {
                'Event': 0,
                'Completed': 0,
                'Timeout': 0
            }
            for hit_file in self.hit_files:
                with ROOT.TFile(str(hit_file), 'read') as f:
                    # Extracts the data from the histograms
                    hist = f.Get("Event/EventCounter")
                    self.__hit_event_counter['Event'] += \
                        hist.GetBinContent(1)
                    self.__hit_event_counter['Completed'] += \
                        hist.GetBinContent(2)
                    self.__hit_event_counter['Timeout'] += \
                        hist.GetBinContent(3)
            for key in self.__hit_event_counter:
                self.__hit_event_counter[key] = int(
                    self.__hit_event_counter[key])
        return self.__hit_event_counter

    @property
    def esd_files(self) -> Iterator[Path]:
        """
        Iterator over the ESD files

        Returns
        -------
        Iterator[Path]
            Iterator over the ESD files
        """
        return self.esd_path.glob('*.root')

    @property
    def esd_event_counter(self) -> Dict[str, int]:
        """
        Number of completed ESD events

        Returns
        -------
        Dict[str, int]
            Dictionary with the number of events
            from esd files. Has the following schema:
            {
                'Event': int,
                'Completed': int
            }
        """
        if self.__esd_event_counter is None:
            self.__esd_event_counter = {
                'Event': 0,
                'Completed': 0
            }
            for esd_file in self.esd_files:
                with ROOT.TFile(str(esd_file), 'read') as f:
                    # Extracts the data from the histograms
                    hist = f.Get("Event/EventCounter")
                    self.__esd_event_counter['Event'] += \
                        hist.GetBinContent(1)
                    self.__esd_event_counter['Completed'] += \
                        hist.GetBinContent(2)
            for key in self.__esd_event_counter:
                self.__esd_event_counter[key] = int(
                    self.__esd_event_counter[key])
        return self.__esd_event_counter

    def get_esd_tchain(self,  n_files: int = -1) -> ROOT.TChain:
        """
        Get a TChain with the ESD files

        Parameters
        ----------
        n_files : int
            Number of files to load.
            If n_files < 0, loads everything

        Returns
        -------
        ROOT.TChain
            TChain with the ESD files
        """
        chain = ROOT.TChain("CollectionTree")
        for i, filename in enumerate(self.esd_files):
            if n_files >= 0 and i >= n_files:
                break
            chain.Add(str(filename))
        return chain

    def get_esd_rdf(self, n_files: int = -1) -> ROOT.RDataFrame:
        """
        Get the RDataFrame for the esd files

        Parameters
        ----------
        n_files : int
            Number of files to load.
            If n_files < 0, loads everything

        Returns
        -------
        ROOT.RDataFrame
            RDataFrame for the esd
        """
        if n_files > 0:
            files = []
            for i, filename in enumerate(self.esd_files):
                if i >= n_files:
                    break
                files.append(str(filename))
        else:
            files = [str(filename) for filename
                     in self.esd_files]
        rdf = ROOT.RDataFrame("CollectionTree", files)
        return rdf

    def get_esd_ak(self,
                   n_files: int = -1,
                   columns: List[str] = None,
                   ak_kwargs: Dict[str, Any] = {}
                   ) -> ak.Array:
        """
        Get the awkward array for the esd files.

        Parameters
        ----------
        n_files : int, optional
            Number of file to load.
            If n_files < 0, loads everything, by default -1
        columns : List[str], optional
            Columns to load, by default None
        ak_kwargs : Dict[str, Any], optional
            Kwargs for awkward.from_rdataframe, by default {}

        Returns
        -------
        ak.Array
            Awkward array for the esd
        """
        esd_rdf = self.get_esd_rdf(n_files)
        esd_ak = ak.from_rdataframe(esd_rdf, columns=columns,
                                    **ak_kwargs)
        for field in esd_ak.fields:
            # The events from EventInfoContainer are always a list
            # with one element, so we extract the first element.
            if field == 'EventInfoContainer_Events.eventNumber':
                esd_ak[field] = ak.values_astype(
                    ak.firsts(esd_ak[field]), np.int32)
            elif field == 'EventInfoContainer_Events.runNumber':
                esd_ak[field] = ak.values_astype(
                    ak.firsts(esd_ak[field]), np.int32)
            elif field.startswith('EventInfoContainer_Events'):
                esd_ak[field] = ak.firsts(esd_ak[field])

    @property
    def aod_files(self) -> Iterator[Path]:
        """
        Iterator over the AOD files

        Returns
        -------
        Iterator[Path]
            Iterator over the AOD files
        """
        return self.aod_path.glob('*.root')

    def get_aod_tchain(self,  n_files: int = -1) -> ROOT.TChain:
        """
        Get a TChain with the AOD files

        Parameters
        ----------
        n_files : int
            Number of files to load.
            If n_files < 0, loads everything

        Returns
        -------
        ROOT.TChain
            TChain with the AOD files
        """
        chain = ROOT.TChain("CollectionTree")
        for i, filename in enumerate(self.aod_files):
            if n_files >= 0 and i >= n_files:
                break
            chain.Add(str(filename))
        return chain

    def get_aod_rdf(self, n_files: int = -1) -> ROOT.RDataFrame:
        """
        Get the RDataFrame for the aod files

        Parameters
        ----------
        n_files : int
            Number of files to load.
            If n_files < 0, loads everything

        Returns
        -------
        ROOT.RDataFrame
            RDataFrame for the aod
        """
        if n_files > 0:
            files = []
            for i, filename in enumerate(self.aod_files):
                if i >= n_files:
                    break
                files.append(str(filename))
        else:
            files = [str(filename) for filename
                     in self.aod_files]
        rdf = ROOT.RDataFrame("CollectionTree", files)
        return rdf

    def get_aod_ak(self,
                   n_files: int = -1,
                   columns: List[str] = None,
                   ak_kwargs: Dict[str, Any] = {}
                   ) -> ak.Array:
        """
        Get the awkward array for the aod files.

        Parameters
        ----------
        n_files : int, optional
            Number of file to load.
            If n_files < 0, loads everything, by default -1
        columns : List[str], optional
            Columns to load, by default None
        ak_kwargs : Dict[str, Any], optional
            Kwargs for awkward.from_rdataframe, by default {}

        Returns
        -------
        ak.Array
            Awkward array for the aod
        """
        aod_rdf = self.get_aod_rdf(n_files)
        return aod_rdf_to_ak(aod_rdf, columns=columns, ak_kwargs=ak_kwargs)

    @property
    def ntuple_files(self) -> Iterator[Path]:
        """
        Iterator over the NTUPLE files

        Returns
        -------
        Iterator[Path]
            Iterator over the NTUPLE files
        """
        return self.ntuple_path.glob('*.root')

    def get_ntuple_rdf(self, n_files: int = -1) -> ROOT.RDataFrame:
        """
        Get the RDataFrame for the ntuple files

        Parameters
        ----------
        n_files : int
            Number of files to load.
            If n_files < 0, loads everything

        Returns
        -------
        ROOT.RDataFrame
            RDataFrame for the ntuple
        """
        if n_files > 0:
            files = []
            for i, filename in enumerate(self.ntuple_files):
                if i >= n_files:
                    break
                files.append(str(filename))
        else:
            files = [str(filename) for filename
                     in self.ntuple_files]
        rdf = ROOT.RDataFrame("events", files)
        return rdf

    def get_ntuple_pdf(self) -> pd.DataFrame:
        """
        Get the pandas DataFrame for the ntuple files

        Returns
        -------
        pd.DataFrame
            DataFrame for the ntuple
        """
        return rdf_to_pandas(self.get_ntuple_rdf())

    def makedirs(self, directory: str) -> Path:
        """
        Create a directory inside the dataset directory

        Parameters
        ----------
        directory : str
            Directory name

        Returns
        -------
        Path
            Absolute path to the created directory
        """
        dir_path = self.path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path.resolve()

    @classmethod
    def from_dir(cls, path: str):
        """
        Create a LztDataset object from a directory

        Parameters
        ----------
        path : str
            Path to the dataset directory

        Returns
        -------
        LztDataset
            LztDataset object

        Raises
        ------
        ValueError
            If the path is not a directory
        """
        if os.path.isdir(path):
            with open(os.path.join(path, 'dataset_info.json'), 'r') as f:
                data = json.load(f)
                return cls(path, **data)
        else:
            raise ValueError(f'{path} is not a directory')
