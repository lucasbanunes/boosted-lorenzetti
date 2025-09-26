import pyarrow as pa
import numpy as np
import typer
from typing import Iterable, List, Any, Dict, Annotated
from pathlib import Path
import ROOT
from collections import defaultdict
import pandas as pd
import duckdb
import logging


from ..utils import open_directories
from . import duckdb as bl_duckdb


PYARROW_SCHEMA = pa.schema([
    ('CaloCellContainer_Cells', pa.list_(pa.struct([
        ('descriptor_link', pa.uint64()),
        ('deta', pa.float32()),
        ('dphi', pa.float32()),
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('phi', pa.float32()),
        ('tau', pa.float32())
    ]))),
    ('CaloClusterContainer_Clusters', pa.list_(pa.struct([
        ('cell_links', pa.list_(pa.uint64())),
        ('deta', pa.float32()),
        ('dphi', pa.float32()),
        ('e', pa.float32()),
        ('e0', pa.float32()),
        ('e1', pa.float32()),
        ('e2', pa.float32()),
        ('e233', pa.float32()),
        ('e237', pa.float32()),
        ('e277', pa.float32()),
        ('e2tsts1', pa.float32()),
        ('e3', pa.float32()),
        ('ehad1', pa.float32()),
        ('ehad2', pa.float32()),
        ('ehad3', pa.float32()),
        ('emaxs1', pa.float32()),
        ('emaxs2', pa.float32()),
        ('eratio', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('etot', pa.float32()),
        ('f0', pa.float32()),
        ('f1', pa.float32()),
        ('f2', pa.float32()),
        ('f3', pa.float32()),
        ('fracMax', pa.float32()),
        ('lambdaCenter', pa.float32()),
        ('lateralMom', pa.float32()),
        ('longitudinalMom', pa.float32()),
        ('phi', pa.float32()),
        ('reta', pa.float32()),
        ('rhad', pa.float32()),
        ('rhad1', pa.float32()),
        ('rphi', pa.float32()),
        ('secondLambda', pa.float32()),
        ('secondR', pa.float32()),
        ('seed_link', pa.int32()),
        ('weta2', pa.float32())
    ]))),
    ('CaloDetDescriptorContainer_Cells', pa.list_(pa.struct([
        ('bc_duration', pa.float32()),
        ('bcid_end', pa.int32()),
        ('bcid_start', pa.int32()),
        ('deta', pa.float32()),
        ('detector', pa.int32()),
        ('dphi', pa.float32()),
        ('e', pa.float32()),
        ('edep', pa.float32()),
        ('edep_per_bunch', pa.list_(pa.float32())),
        ('eta', pa.float32()),
        ('hash', pa.uint64()),
        ('phi', pa.float32()),
        ('pulse', pa.list_(pa.float32())),
        ('sampling', pa.int32()),
        ('tau', pa.float32()),
        ('tof', pa.list_(pa.float32())),
        ('z', pa.float32())
    ]))),
    ('CaloRingsContainer_Rings', pa.list_(pa.struct([
        ('cluster_link', pa.int32()),
        ('rings', pa.list_(pa.float32()))
    ]))),
    ('ElectronContainer_Electrons', pa.list_(pa.struct([
        ('cluster_link', pa.int32()),
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('phi', pa.float32()),
        ('isEM', pa.list_(pa.bool_())),
    ]))),
    ('EventInfoContainer_Events', pa.list_(pa.struct([
        ('avgmu', pa.float32()),
        ('eventNumber', pa.float32()),
        ('runNumber', pa.float32()),
        # ('totmu', pa.float32())
    ]))),
    ('SeedContainer_Seeds', pa.list_(pa.struct([
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('id', pa.int32()),
        ('phi', pa.float32())
    ]))),
    ('TruthParticleContainer_Particles', pa.list_(pa.struct([
        ('e', pa.float32()),
        ('et', pa.float32()),
        ('eta', pa.float32()),
        ('pdgid', pa.int32()),
        ('phi', pa.float32()),
        ('px', pa.float32()),
        ('py', pa.float32()),
        ('pz', pa.float32()),
        ('seedid', pa.int32()),
        ('vx', pa.float32()),
        ('vy', pa.float32()),
        ('vz', pa.float32())
    ])))
])

CLUSTERS_PYARROW_SCHEMA = pa.schema([
    ('id', pa.int64()),
    ('event_id', pa.int64()),
    ('deta', pa.float32()),
    ('dphi', pa.float32()),
    ('e', pa.float32()),
    ('e0', pa.float32()),
    ('e1', pa.float32()),
    ('e2', pa.float32()),
    ('e233', pa.float32()),
    ('e237', pa.float32()),
    ('e277', pa.float32()),
    ('e2tsts1', pa.float32()),
    ('e3', pa.float32()),
    ('ehad1', pa.float32()),
    ('ehad2', pa.float32()),
    ('ehad3', pa.float32()),
    ('emaxs1', pa.float32()),
    ('emaxs2', pa.float32()),
    ('eratio', pa.float32()),
    ('et', pa.float32()),
    ('eta', pa.float32()),
    ('etot', pa.float32()),
    ('f0', pa.float32()),
    ('f1', pa.float32()),
    ('f2', pa.float32()),
    ('f3', pa.float32()),
    ('fracMax', pa.float32()),
    ('lambdaCenter', pa.float32()),
    ('lateralMom', pa.float32()),
    ('longitudinalMom', pa.float32()),
    ('phi', pa.float32()),
    ('reta', pa.float32()),
    ('rhad', pa.float32()),
    ('rhad1', pa.float32()),
    ('rphi', pa.float32()),
    ('secondLambda', pa.float32()),
    ('secondR', pa.float32()),
    ('seed_link', pa.int32()),
    ('weta2', pa.float32()),
    ('rings', pa.list_(pa.float32()))
])

STRUCTS = [
    'CaloCellContainer_Cells',
    'CaloClusterContainer_Clusters',
    'CaloDetDescriptorContainer_Cells',
    'CaloRingsContainer_Rings',
    'ElectronContainer_Electrons',
    'EventInfoContainer_Events',
    'SeedContainer_Seeds',
    'TruthParticleContainer_Particles'
]

CREATE_EVENT_INFO_TABLE_QUERY = """
CREATE TABLE events (
    id UBIGINT PRIMARY KEY,
    avgmu FLOAT,
    eventNumber FLOAT,
    runNumber FLOAT
);
"""

CREATE_SEEDS_TABLE_QUERY = """
CREATE TABLE seeds (
    id UBIGINT PRIMARY KEY,
    event_id UBIGINT REFERENCES events(id),
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    sid INTEGER,
    phi FLOAT
);
"""

CREATE_CLUSTERS_TABLE_QUERY = """
CREATE TABLE clusters (
    id UBIGINT PRIMARY KEY,
    event_id UBIGINT REFERENCES events(id),
    deta FLOAT,
    dphi FLOAT,
    e FLOAT,
    e0 FLOAT,
    e1 FLOAT,
    e2 FLOAT,
    e233 FLOAT,
    e237 FLOAT,
    e277 FLOAT,
    e2tsts1 FLOAT,
    e3 FLOAT,
    ehad1 FLOAT,
    ehad2 FLOAT,
    ehad3 FLOAT,
    emaxs1 FLOAT,
    emaxs2 FLOAT,
    eratio FLOAT,
    et FLOAT,
    eta FLOAT,
    etot FLOAT,
    f0 FLOAT,
    f1 FLOAT,
    f2 FLOAT,
    f3 FLOAT,
    fracMax FLOAT,
    lambdaCenter FLOAT,
    lateralMom FLOAT,
    longitudinalMom FLOAT,
    phi FLOAT,
    reta FLOAT,
    rhad FLOAT,
    rhad1 FLOAT,
    rphi FLOAT,
    secondLambda FLOAT,
    secondR FLOAT,
    seed_id UBIGINT REFERENCES seeds(id),
    weta2 FLOAT,
    rings FLOAT[]
);
"""


CREATE_CALO_CELLS_TABLE_QUERY = """
CREATE TABLE calo_cells (
    id UBIGINT PRIMARY KEY,
    descriptor_id UBIGINT REFERENCES calo_descriptor_cells(id),
    deta FLOAT,
    dphi FLOAT,
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    phi FLOAT,
    tau FLOAT,
);
"""

CREATE_CALO_DESCRIPTOR_CELLS_TABLE_QUERY = """
CREATE TABLE calo_descriptor_cells (
    id UBIGINT PRIMARY KEY,
    cluster_id UBIGINT REFERENCES clusters(id),
    bc_duration FLOAT,
    bcid_end INTEGER,
    bcid_start INTEGER,
    deta FLOAT,
    detector INTEGER,
    dphi FLOAT,
    e FLOAT,
    edep FLOAT,
    edep_per_bunch FLOAT[],
    eta FLOAT,
    hash UBIGINT,
    phi FLOAT,
    pulse FLOAT[],
    sampling INTEGER,
    tau FLOAT,
    tof FLOAT[],
    z FLOAT
);
"""

CREATE_ELECTRONS_TABLE_QUERY = """
CREATE TABLE electrons (
    id UBIGINT PRIMARY KEY,
    cluster_id UBIGINT REFERENCES clusters(id),
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    phi FLOAT,
    isEM BOOLEAN[]
);
"""

CREATE_TRUTH_PARTICLES_TABLE_QUERY = """
CREATE TABLE truth_particles (
    id UBIGINT PRIMARY KEY,
    event_id UBIGINT REFERENCES events(id),
    seed_id UBIGINT REFERENCES seeds(id),
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    pdgid INTEGER,
    phi FLOAT,
    px FLOAT,
    py FLOAT,
    pz FLOAT,
    vx FLOAT,
    vy FLOAT,
    vz FLOAT
);
"""


def EventInfoContainer_Events_as_python(data):
    """
    Convert EventInfoContainer_Events data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw event info data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted event info data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'avgmu': d.avgmu,
            'eventNumber': d.eventNumber,
            'runNumber': d.runNumber})
    return new_data


def CaloCellContainer_Cells_as_python(data):
    """
    Convert CaloCellContainer_Cells data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw calorimeter cell data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted calorimeter cell data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'descriptor_link': d.descriptor_link,
            'deta': d.deta,
            'dphi': d.dphi,
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'phi': d.phi,
            'tau': d.tau})
    return new_data


def CaloClusterContainer_Clusters_as_python(data):
    """
    Convert CaloClusterContainer_Clusters data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw calorimeter cluster data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted calorimeter cluster data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'cell_links': np.array(d.cell_links, dtype=np.uint64).tolist(),
            'deta': d.deta,
            'dphi': d.dphi,
            'e': d.e,
            'e0': d.e0,
            'e1': d.e1,
            'e2': d.e2,
            'e233': d.e233,
            'e237': d.e237,
            'e277': d.e277,
            'e2tsts1': d.e2tsts1,
            'e3': d.e3,
            'ehad1': d.ehad1,
            'ehad2': d.ehad2,
            'ehad3': d.ehad3,
            'emaxs1': d.emaxs1,
            'emaxs2': d.emaxs2,
            'eratio': d.eratio,
            'et': d.et,
            'eta': d.eta,
            'etot': d.etot,
            'f0': d.f0,
            'f1': d.f1,
            'f2': d.f2,
            'f3': d.f3,
            'fracMax': d.fracMax,
            'lambdaCenter': d.lambdaCenter,
            'lateralMom': d.lateralMom,
            'longitudinalMom': d.longitudinalMom,
            'phi': d.phi,
            'reta': d.reta,
            'rhad': d.rhad,
            'rhad1': d.rhad1,
            'rphi': d.rphi,
            'secondLambda': d.secondLambda,
            'secondR': d.secondR,
            'seed_link': d.seed_link,
            'weta2': d.weta2})
    return new_data


def CaloDetDescriptorContainer_Cells_as_python(data):
    """
    Convert CaloDetDescriptorContainer_Cells data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw calorimeter detector descriptor cell data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted calorimeter detector descriptor cell data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'bc_duration': d.bc_duration,
            'bcid_end': d.bcid_end,
            'bcid_start': d.bcid_start,
            'deta': d.deta,
            'detector': d.detector,
            'dphi': d.dphi,
            'e': d.e,
            'edep': d.edep,
            'edep_per_bunch': np.array(d.edep_per_bunch).tolist(),
            'eta': d.eta,
            'hash': d.hash,
            'phi': d.phi,
            'pulse': np.array(d.pulse).tolist(),
            'sampling': d.sampling,
            'tau': d.tau,
            'tof': np.array(d.tof).tolist(),
            'z': d.z})
    return new_data


def CaloRingsContainer_Rings_as_python(data):
    """
    Convert CaloRingsContainer_Rings data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw calorimeter rings data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted calorimeter rings data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'cluster_link': d.cluster_link,
            'rings': np.array(d.rings).tolist()})
    return new_data


def ElectronContainer_Electrons_as_python(data):
    """
    Convert ElectronContainer_Electrons data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw electron data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted electron data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'cluster_link': d.cluster_link,
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'phi': d.phi,
            'isEM': np.array(d.isEM).tolist()})
    return new_data


def SeedContainer_Seeds_values_as_python(data):
    """
    Convert SeedContainer_Seeds data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw seed data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted seed data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'id': d.id,
            'phi': d.phi})
    return new_data


def TruthParticleContainer_Particles_as_python(data):
    """
    Convert TruthParticleContainer_Particles data to Python dictionary format.

    Parameters
    ----------
    data : Iterable
        Raw truth particle data from ROOT file.

    Returns
    -------
    List[Dict[str, Any]]
        Converted truth particle data in Python dictionary format.
    """
    new_data = []
    for d in data:
        new_data.append({
            'e': d.e,
            'et': d.et,
            'eta': d.eta,
            'pdgid': d.pdgid,
            'phi': d.phi,
            'px': d.px,
            'py': d.py,
            'pz': d.pz,
            'seedid': d.seedid,
            'vx': d.vx,
            'vy': d.vy,
            'vz': d.vz})
    return new_data


def event_as_python(event):
    """
    Convert a single AOD event to Python dictionary format.

    Parameters
    ----------
    event : ROOT.TTree.event
        A single event from the ROOT TTree.

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        The event data converted to Python dictionary format with all containers.
    """
    new_data = {
        'CaloCellContainer_Cells': CaloCellContainer_Cells_as_python(event.CaloCellContainer_Cells),
        'CaloClusterContainer_Clusters': CaloClusterContainer_Clusters_as_python(event.CaloClusterContainer_Clusters),
        'CaloDetDescriptorContainer_Cells': CaloDetDescriptorContainer_Cells_as_python(event.CaloDetDescriptorContainer_Cells),
        'CaloRingsContainer_Rings': CaloRingsContainer_Rings_as_python(event.CaloRingsContainer_Rings),
        'ElectronContainer_Electrons': ElectronContainer_Electrons_as_python(event.ElectronContainer_Electrons),
        'EventInfoContainer_Events': EventInfoContainer_Events_as_python(event.EventInfoContainer_Events),
        'SeedContainer_Seeds': SeedContainer_Seeds_values_as_python(event.SeedContainer_Seeds),
        'TruthParticleContainer_Particles': TruthParticleContainer_Particles_as_python(event.TruthParticleContainer_Particles)

    }
    return new_data


app = typer.Typer(
    name='aod',
    help="AOD file conversion utilities",
)


def sample_generator(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
                     ttree_name: str = 'CollectionTree'):
    """
    Generate AOD events from ROOT files as Python dictionaries.

    Parameters
    ----------
    input_file : str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain
        Path to input ROOT file(s) or an existing ROOT TChain.
    ttree_name : str, optional
        Name of the TTree to read from the ROOT file. Default is 'CollectionTree'.

    Yields
    ------
    Dict[str, List[Dict[str, Any]]]
        Individual AOD events converted to Python dictionary format.
    """
    if not isinstance(input_file, ROOT.TChain):
        chain = ROOT.TChain(ttree_name)
        for file in open_directories(input_file, file_ext='root'):
            chain.Add(str(file))
    for event in chain:
        logging.debug(f'Processing event {event.EventInfoContainer_Events[0].eventNumber}')
        yield event_as_python(event)


def to_dict(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
            ttree_name: str = 'CollectionTree'):
    """
    Convert a single AOD root file to a dictionary representation.

    Parameters
    ----------
    input_file : str | Path
        The path to the input AOD root file.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary representation of the AOD data.
    """

    data = defaultdict(list)
    for event in sample_generator(input_file, ttree_name):
        for col_name, value in event.items():
            logging.debug(f'Appending {col_name} with {value}')
            data[col_name].append(value)
    return data


def to_pdf(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
           ttree_name: str = 'CollectionTree',) -> pd.DataFrame:
    """
    Convert a single AOD root file to a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain
        The path to the input AOD root file(s) or an existing ROOT TChain.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the AOD data with proper PyArrow dtypes.
    """
    data = to_dict(input_file, ttree_name)
    data = pd.DataFrame.from_dict(data)
    logging.debug(f'Data keys: {data.keys()}')
    for field_name in PYARROW_SCHEMA.names:
        data[field_name] = data[field_name].astype(
            pd.ArrowDtype(PYARROW_SCHEMA.field(field_name).type))
    return data


@app.command(
    help='Converts AOD files to parquet files'
)
def to_parquet(
        input_file: Annotated[
            str,
            typer.Option(help="Path to the input AOD file.")
        ],
        output_file: Annotated[
            str,
            typer.Option(help="Path to the output parquet file.")
        ],
        ttree_name: Annotated[
            str,
            typer.Option(help="NTuple Tree name inside the .root file")
        ] = 'CollectionTree') -> None:
    """
    Convert AOD root files to a AOD parquet file.

    Parameters
    ----------
    input_file : str
        The path to the input AOD file.
    output_file : str
        The path where the output parquet file will be saved.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.
    """
    df = to_pdf(input_file, ttree_name)
    df.to_parquet(output_file, index=False, compression='gzip')


OutputFileOption = Annotated[
    Path,
    typer.Option(help='Path to the output file.')
]

DescriptionOption = Annotated[
    str,
    typer.Option(help='Description of the resulting data.')
]


@app.command(
    help='Converts AOD root files to duckdb'
)
def to_duckdb(
    input_file: Annotated[
        List[str],
        typer.Option(help='Path to the input AOD file(s).')
    ],
    output_file: OutputFileOption,
    ttree_name: str = 'CollectionTree',
    batch_size: int = -1,
    description: DescriptionOption = None
) -> None:
    """
    Convert AOD root files to a duckdb database.

    Parameters
    ----------
    input_file : List[str]
        The path to the input AOD file.
    output_file : str
        The path where the output duckdb table will be saved.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.
    batch_size: int , optional
        The size of the batches to process. Default is -1 (all data at once).
    """
    with duckdb.connect(str(output_file)) as conn:
        conn.execute(CREATE_EVENT_INFO_TABLE_QUERY)
        conn.execute(CREATE_SEEDS_TABLE_QUERY)
        conn.execute(CREATE_CLUSTERS_TABLE_QUERY)
        conn.execute(CREATE_CALO_DESCRIPTOR_CELLS_TABLE_QUERY)
        conn.execute(CREATE_CALO_CELLS_TABLE_QUERY)
        conn.execute(CREATE_ELECTRONS_TABLE_QUERY)
        conn.execute(CREATE_TRUTH_PARTICLES_TABLE_QUERY)

    events = defaultdict(list)

    clusters = defaultdict(list)
    cluster_id_counter = 1

    seeds = defaultdict(list)
    seed_id_counter = 1

    calo_descriptor_cells = defaultdict(list)
    calo_descriptor_id_counter = 1

    calo_cells = defaultdict(list)
    calo_cell_id_counter = 1

    electrons = defaultdict(list)
    electron_id_counter = 1

    truth_particles = defaultdict(list)
    truth_particles_counter = 1

    if batch_size < 0:
        batch_size = np.inf
    batch_counter = 0
    batch_samples = 0
    logging.info('Starting batch processing')
    for event_id, aod_event in enumerate(sample_generator(input_file, ttree_name),
                                         start=1):

        if len(aod_event['EventInfoContainer_Events']) != 1:
            raise RuntimeError("Expected exactly one event")
        events['id'].append(event_id)
        for key, value in aod_event['EventInfoContainer_Events'][0].items():
            events[key].append(value)

        seed_link_map = {}
        seed_event_id_seed_id_map = {}
        for i, seed_struct in enumerate(aod_event['SeedContainer_Seeds']):
            seed_link_map[i] = seed_id_counter
            seeds['id'].append(seed_id_counter)
            seeds['event_id'].append(event_id)
            for key, value in seed_struct.items():
                if key == 'id':
                    seed_event_id_seed_id_map[value] = seed_id_counter
                else:
                    seeds[key].append(value)
            seed_id_counter += 1

        cluster_link_id_map = {}
        cell_cluster_id_map = {}
        for i, cluster_struct in enumerate(aod_event['CaloClusterContainer_Clusters']):
            clusters['id'].append(cluster_id_counter)
            cluster_link_id_map[i] = cluster_id_counter
            clusters['event_id'].append(event_id)
            for key, value in cluster_struct.items():
                if key == 'cell_links':
                    for cell_link in value:
                        cell_cluster_id_map[cell_link] = cluster_id_counter
                elif key == 'seed_link':
                    clusters['seed_id'].append(seed_link_map[value])
                else:
                    clusters[key].append(value)
            clusters['rings'].append([])
            for ring_struct in aod_event['CaloRingsContainer_Rings']:
                if ring_struct['cluster_link'] == i:
                    clusters['rings'][-1] = ring_struct['rings']
                    break
            cluster_id_counter += 1

        descriptor_link_hash_id_map = {}
        for i, calo_descriptor_cell_struct in enumerate(aod_event['CaloDetDescriptorContainer_Cells']):
            calo_descriptor_cells['id'].append(calo_descriptor_id_counter)
            for key, value in calo_descriptor_cell_struct.items():
                if key == 'hash':
                    calo_descriptor_cells['cluster_id'].append(
                        cell_cluster_id_map[value])
                    descriptor_link_hash_id_map[value] = calo_descriptor_id_counter
                else:
                    calo_descriptor_cells[key].append(value)
            calo_descriptor_id_counter += 1

        for i, calo_cells_struct in enumerate(aod_event['CaloCellContainer_Cells']):
            calo_cells['id'].append(calo_cell_id_counter)
            for key, value in calo_cells_struct.items():
                if key == 'descriptor_link':
                    calo_cells['descriptor_id'].append(
                        descriptor_link_hash_id_map[value])
                else:
                    calo_cells[key].append(value)
            calo_cell_id_counter += 1

        for i, electrons_struct in enumerate(aod_event['ElectronContainer_Electrons']):
            electrons['id'].append(electron_id_counter)
            for key, value in electrons_struct.items():
                if key == 'cluster_link':
                    key = 'cluster_id'
                    if value not in cluster_link_id_map:
                        logging.warning(
                            f'Event {event_id} electron {i} does not have a valid cluster_link: {value} (Clusters: {cluster_link_id_map})')
                        value = None
                    else:
                        value = cluster_link_id_map[value]
                electrons[key].append(value)
            electron_id_counter += 1

        for i, truth_particle_struct in enumerate(aod_event['TruthParticleContainer_Particles']):
            truth_particles['id'].append(truth_particles_counter)
            truth_particles['event_id'].append(event_id)
            for key, value in truth_particle_struct.items():
                if key == 'seedid':
                    truth_particles['seed_id'].append(
                        seed_event_id_seed_id_map[value])
                else:
                    truth_particles[key].append(value)
            truth_particles_counter += 1

        batch_samples += 1
        if batch_samples >= batch_size:
            logging.info(
                f'Inserting batch {batch_counter} with {len(events["id"])} events')
            with duckdb.connect(str(output_file)) as conn:
                write_batch_to_duck_db(conn,
                                       events,
                                       clusters,
                                       seeds,
                                       calo_descriptor_cells,
                                       calo_cells,
                                       electrons,
                                       truth_particles)
            events = defaultdict(list)
            clusters = defaultdict(list)
            seeds = defaultdict(list)
            calo_descriptor_cells = defaultdict(list)
            calo_cells = defaultdict(list)
            electrons = defaultdict(list)
            truth_particles = defaultdict(list)
            batch_counter += 1
            batch_samples = 0

    # Dumps remaining data
    logging.info('Inserting remaining data')
    with duckdb.connect(str(output_file)) as conn:
        write_batch_to_duck_db(conn,
                               events,
                               clusters,
                               seeds,
                               calo_descriptor_cells,
                               calo_cells,
                               electrons,
                               truth_particles)
        logging.debug('Finished batch processing')
        bl_duckdb.add_metadata_table(conn,
                                     name=output_file.stem.replace(
                                         '-', '_').replace('.', '_'),
                                     description=description)
    logging.info(f'DuckDB saved to {output_file}')


def write_batch_to_duck_db(
        conn: duckdb.DuckDBPyConnection,
        events: Dict[str, List[Any]],
        clusters: Dict[str, List[Any]],
        seeds: Dict[str, List[Any]],
        calo_descriptor_cells: Dict[str, List[Any]],
        calo_cells: Dict[str, List[Any]],
        electrons: Dict[str, List[Any]],
        truth_particles: Dict[str, List[Any]]
) -> None:
    """
    Write a batch of events and clusters to the DuckDB database.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        The connection to the DuckDB database.
    events : Dict[str, List[Any]]
        The events data to write.
    clusters : Dict[str, List[Any]]
        The clusters data to write.
    seeds : Dict[str, List[Any]]
        The seeds data to write.
    calo_descriptor_cells : Dict[str, List[Any]]
        The calorimeter descriptor cells data to write.
    calo_cells : Dict[str, List[Any]]
        The calorimeter cells data to write.
    electrons : Dict[str, List[Any]]
        The electrons data to write.
    truth_particles : Dict[str, List[Any]]
        The truth particles data to write.

    Returns
    -------
    None
    """
    events_df = pd.DataFrame.from_dict(events)  # noqa: F841 Ignores the unused variable
    if len(events_df):
        conn.execute("INSERT INTO events BY NAME SELECT * FROM events_df;")

    seeds_df = pd.DataFrame.from_dict(seeds)  # noqa: F841 Ignores the unused variable
    if len(seeds_df):
        conn.execute("INSERT INTO seeds BY NAME SELECT * FROM seeds_df;")

    clusters_df = pd.DataFrame.from_dict(clusters)
    if len(clusters_df):
        for field_name in CLUSTERS_PYARROW_SCHEMA.names:
            if field_name == 'seed_link':
                clusters_df['seed_id'] = clusters_df['seed_id'].astype(
                    pd.ArrowDtype(CLUSTERS_PYARROW_SCHEMA.field('seed_link').type))
            else:
                clusters_df[field_name] = clusters_df[field_name].astype(
                    pd.ArrowDtype(CLUSTERS_PYARROW_SCHEMA.field(field_name).type))
        conn.execute("INSERT INTO clusters BY NAME SELECT * FROM clusters_df;")

    calo_descriptor_cells_df = pd.DataFrame.from_dict(calo_descriptor_cells)  # noqa: F841 Ignores the unused variable
    if len(calo_descriptor_cells_df):
        conn.execute(
            "INSERT INTO calo_descriptor_cells BY NAME SELECT * FROM calo_descriptor_cells_df;")

    calo_cells_df = pd.DataFrame.from_dict(calo_cells)  # noqa: F841 Ignores the unused variable
    if len(calo_cells_df):
        conn.execute(
            "INSERT INTO calo_cells BY NAME SELECT * FROM calo_cells_df;")

    electrons_df = pd.DataFrame.from_dict(electrons)  # noqa: F841 Ignores the unused variable
    if len(electrons_df):
        conn.execute(
            "INSERT INTO electrons BY NAME SELECT * FROM electrons_df;")

    truth_particles_df = pd.DataFrame.from_dict(truth_particles)  # noqa: F841 Ignores the unused variable
    if len(truth_particles_df):
        conn.execute(
            "INSERT INTO truth_particles BY NAME SELECT * FROM truth_particles_df;")


CREATE_RINGER_DATABASE_SOURCES_TABLE_QUERY = """
CREATE SEQUENCE source_id_seq START 1;
CREATE TABLE sources (
    id UBIGINT PRIMARY KEY DEFAULT nextval('source_id_seq'),
    name TEXT NOT NULL,
    description TEXT,
    label UTINYINT NOT NULL
);
"""

CREATE_RINGER_DATASET_TABLE_QUERY = """
CREATE SEQUENCE data_id_seq START 1;
CREATE TABLE data (
    id UBIGINT PRIMARY KEY DEFAULT nextval('data_id_seq'),
    source_id UBIGINT REFERENCES sources(id),
    event_id UBIGINT,
    eventNumber FLOAT,
    avgmu FLOAT,
    cluster_id UBIGINT,
    cl_eta FLOAT,
    cl_phi FLOAT,
    cl_e FLOAT,
    cl_et FLOAT,
    cl_deta FLOAT,
    cl_dphi FLOAT,
    cl_e0 FLOAT,
    cl_e1 FLOAT,
    cl_e2 FLOAT,
    cl_e3 FLOAT,
    cl_ehad1 FLOAT,
    cl_ehad2 FLOAT,
    cl_ehad3 FLOAT,
    cl_etot FLOAT,
    cl_e233 FLOAT,
    cl_e237 FLOAT,
    cl_e277 FLOAT,
    cl_emaxs1 FLOAT,
    cl_emaxs2 FLOAT,
    cl_e2tsts1 FLOAT,
    cl_reta FLOAT,
    cl_rphi FLOAT,
    cl_rhad FLOAT,
    cl_rhad1 FLOAT,
    cl_eratio FLOAT,
    cl_f0 FLOAT,
    cl_f1 FLOAT,
    cl_f2 FLOAT,
    cl_f3 FLOAT,
    cl_weta2 FLOAT,
    cl_rings FLOAT[],
    cl_secondR FLOAT,
    cl_lambdaCenter FLOAT,
    cl_fracMax FLOAT,
    cl_lateralMom FLOAT,
);
"""

SELECT_RINGER_DATASET_DATA_TABLE_QUERY = """
SELECT
    {source_id} as source_id,
    {db_name}.events.id as event_id,
    {db_name}.events.eventNumber as eventNumber,
    {db_name}.events.avgmu as avgmu,
    {db_name}.clusters.id as cluster_id,
    {db_name}.clusters.eta as cl_eta,
    {db_name}.clusters.phi as cl_phi,
    {db_name}.clusters.e as cl_e,
    {db_name}.clusters.et as cl_et,
    {db_name}.clusters.deta as cl_deta,
    {db_name}.clusters.dphi as cl_dphi,
    {db_name}.clusters.e0 as cl_e0,
    {db_name}.clusters.e1 as cl_e1,
    {db_name}.clusters.e2 as cl_e2,
    {db_name}.clusters.e3 as cl_e3,
    {db_name}.clusters.ehad1 as cl_ehad1,
    {db_name}.clusters.ehad2 as cl_ehad2,
    {db_name}.clusters.ehad3 as cl_ehad3,
    {db_name}.clusters.etot as cl_etot,
    {db_name}.clusters.e233 as cl_e233,
    {db_name}.clusters.e237 as cl_e237,
    {db_name}.clusters.e277 as cl_e277,
    {db_name}.clusters.emaxs1 as cl_emaxs1,
    {db_name}.clusters.emaxs2 as cl_emaxs2,
    {db_name}.clusters.e2tsts1 as cl_e2tsts1,
    {db_name}.clusters.reta as cl_reta,
    {db_name}.clusters.rphi as cl_rphi,
    {db_name}.clusters.rhad as cl_rhad,
    {db_name}.clusters.rhad1 as cl_rhad1,
    {db_name}.clusters.eratio as cl_eratio,
    {db_name}.clusters.f0 as cl_f0,
    {db_name}.clusters.f1 as cl_f1,
    {db_name}.clusters.f2 as cl_f2,
    {db_name}.clusters.f3 as cl_f3,
    {db_name}.clusters.weta2 as cl_weta2,
    {db_name}.clusters.rings as cl_rings,
    {db_name}.clusters.secondR as cl_secondR,
    {db_name}.clusters.lambdaCenter as cl_lambdaCenter,
    {db_name}.clusters.fracMax as cl_fracMax,
    {db_name}.clusters.lateralMom as cl_lateralMom
FROM {db_name}.events
    LEFT JOIN {db_name}.clusters ON {db_name}.events.id = {db_name}.clusters.event_id;
"""


@app.command(
    help='Creates a Ringer dataset from AOD duckdb'
)
def create_ringer_dataset(
    input_dbs: Annotated[
        str,
        typer.Option(
            help='Comma separated list of paths to the input AOD duckdb file(s).')
    ],
    labels: Annotated[
        str,
        typer.Option(
            help='Comma separated list of positive class labels "0, 1, 3"')
    ],
    output_file: OutputFileOption,
    description: DescriptionOption = None
) -> None:
    """
    Create a Ringer dataset from a set of AOD duckdb databases.

    Parameters
    ----------
    input_dbs : List[Path]
        The paths to the input AOD duckdb files.
    output_db : Path
        The path where the output Ringer dataset duckdb file will be saved.
    db_name : str, optional
        The name of the database inside the duckdb file. Default is 'aod'.
    description : str, optional
        Description of the resulting data.
    """
    if isinstance(input_dbs, str):
        input_dbs = [Path(db.strip()) for db in input_dbs.split(',')]
    if isinstance(labels, str):
        labels = [int(label.strip()) for label in labels.split(',')]
    with duckdb.connect(str(output_file)) as conn:
        logging.info(f'Creating Ringer dataset duckdb at {output_file}')
        conn.execute(CREATE_RINGER_DATABASE_SOURCES_TABLE_QUERY)
        conn.execute(CREATE_RINGER_DATASET_TABLE_QUERY)
        for source_id, (label, input_db) in enumerate(zip(labels, input_dbs), start=1):

            if not input_db.exists():
                raise FileNotFoundError(
                    f'Input duckdb file {input_db} does not exist')
            with duckdb.connect(str(input_db)) as input_conn:
                metadata = bl_duckdb.get_metadata(input_conn)

            metadata['name'] = input_db.stem.replace(
                '-', '_').replace('.', '_')
            conn.execute("INSERT INTO sources (id, name, description, label) VALUES (?, ?, ?, ?);",
                         (source_id, metadata['name'], metadata['description'], label))

            logging.info(
                f'Processing label {label} from {input_db} as database {metadata["name"]}')
            conn.execute(
                f"ATTACH DATABASE '{input_db}' AS {metadata['name']};")
            formated_select_query = SELECT_RINGER_DATASET_DATA_TABLE_QUERY.format(
                source_id=source_id, db_name=metadata['name'])
            conn.execute(f"INSERT INTO data BY NAME {formated_select_query};")

        bl_duckdb.add_metadata_table(conn,
                                     name=output_file.stem.replace(
                                         '-', '_').replace('.', '_'),
                                     description=description)
        logging.info(f'Ringer dataset duckdb saved to {output_file}')
