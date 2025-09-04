import pyarrow as pa
import numpy as np
import typer
from typing import Iterable, List, Any, Dict, Annotated, Generator
from pathlib import Path
import ROOT
from collections import defaultdict
import pandas as pd
import duckdb
import logging


from ..utils import open_directories


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
    id BIGINT PRIMARY KEY DEFAULT 0,
    avgmu FLOAT,
    eventNumber FLOAT,
    runNumber FLOAT
);
"""

CREATE_CLUSTERS_TABLE_QUERY = """
CREATE TABLE clusters (
    id BIGINT PRIMARY KEY DEFAULT 0,
    event_id BIGINT REFERENCES events(id),
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
    seed_link INTEGER,
    weta2 FLOAT,
    rings FLOAT[]
);
"""


CREATE_CALO_CELLS_TABLE_QUERY = """
CREATE TABLE calo_cells (
    id INTEGER PRIMARY KEY DEFAULT 0,
    cluster_id BIGINT REFERENCES clusters(id),
    deta FLOAT,
    dphi FLOAT,
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    phi FLOAT,
    tau FLOAT,
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
    hash BIGINT,
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
    id BIGINT PRIMARY KEY DEFAULT 0,
    cluster_id BIGINT REFERENCES clusters(id),
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    phi FLOAT,
    isEM BOOLEAN[]
);
"""

CREATE_SEEDS_TABLE_QUERY = """
CREATE TABLE seeds (
    id BIGINT PRIMARY KEY DEFAULT 0,
    event_id BIGINT REFERENCES events(id),
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    id INTEGER,
    phi FLOAT
);
"""

CREATE_TRUTH_PARTICLES_TABLE_QUERY = """
CREATE TABLE truth_particles (
    id BIGINT PRIMARY KEY DEFAULT 0,
    event_id BIGINT REFERENCES events(id),
    e FLOAT,
    et FLOAT,
    eta FLOAT,
    pdgid INTEGER,
    phi FLOAT,
    px FLOAT,
    py FLOAT,
    pz FLOAT,
    seedid INTEGER,
    vx FLOAT,
    vy FLOAT,
    vz FLOAT
);
"""


def EventInfoContainer_Events_as_python(data):
    new_data = []
    for d in data:
        new_data.append({
            'avgmu': d.avgmu,
            'eventNumber': d.eventNumber,
            'runNumber': d.runNumber})
    return new_data


def CaloCellContainer_Cells_as_python(data):
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
            # seed_link is an int32, but it is a list in the schema
            'seed_link': d.seed_link,
            'weta2': d.weta2})
    return new_data


def CaloDetDescriptorContainer_Cells_as_python(data):
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
    new_data = []
    for d in data:
        new_data.append({
            'cluster_link': d.cluster_link,
            'rings': np.array(d.rings).tolist()})
    return new_data


def ElectronContainer_Electrons_as_python(data):
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
    if not isinstance(input_file, ROOT.TChain):
        chain = ROOT.TChain(ttree_name)
        for file in open_directories(input_file, file_ext='root'):
            chain.Add(str(file))
    for event in chain:
        yield event_as_python(event)


def to_dict(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
            ttree_name: str = 'CollectionTree',
            batch_size: int = -1):
    """
    Convert a single AOD root file to a dictionary representation.

    Parameters
    ----------
    input_file : str | Path
        The path to the input AOD root file.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.
    batch_size: int, optional
        Defines the size of the batch to process. Default is -1 (all data at once).

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary representation of the AOD data.
    """

    data = defaultdict(list)
    if batch_size < 0:
        batch_size = np.inf
    batch_counter = 0
    for event in sample_generator(input_file, ttree_name):
        for col_name, value in event.items():
            data[col_name].append(value)
        batch_counter += 1
        if batch_counter >= batch_size:
            yield data
            data = defaultdict(list)
            batch_counter = 0
    return data


def to_pdf(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
           ttree_name: str = 'CollectionTree',) -> pd.DataFrame:
    """
    Convert a single AOD root file to a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path
        The path to the input AOD root file.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'CollectionTree'.
    """
    data = to_dict(input_file, ttree_name)
    data = pd.DataFrame.from_dict(data)
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


@app.command(
    help='Converts AOD root files to duckdb'
)
def to_duckdb(
    input_file: Annotated[
        List[str],
        typer.Option(help='Path to the input AOD file(s).')
    ],
    output_file: Annotated[
        str,
        typer.Option(help='Path to the output duckdb file.')
    ],
    ttree_name: str = 'CollectionTree',
    batch_size: int = -1
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
        conn.execute(CREATE_CLUSTERS_TABLE_QUERY)
        # conn.execute(CREATE_CALO_CELLS_TABLE_QUERY)
        # conn.execute(CREATE_ELECTRONS_TABLE_QUERY)
        # conn.execute(CREATE_SEEDS_TABLE_QUERY)
        # conn.execute(CREATE_TRUTH_PARTICLES_TABLE_QUERY)

        events = defaultdict(list)

        clusters = defaultdict(list)
        cluster_id_counter = 0

        if batch_size < 0:
            batch_size = np.inf
        batch_counter = 0
        batch_samples = 0
        logging.info('Starting batch processing')
        for event_id, aod_event in enumerate(sample_generator(input_file, ttree_name)):

            if len(aod_event['EventInfoContainer_Events']) != 1:
                raise RuntimeError("Expected exactly one event")
            events['id'].append(event_id)
            for key, value in aod_event['EventInfoContainer_Events'][0].items():
                events[key].append(value)

            for i, cluster_struct in enumerate(aod_event['CaloClusterContainer_Clusters']):
                clusters['id'].append(cluster_id_counter)
                clusters['event_id'].append(event_id)
                for key, value in cluster_struct.items():
                    if key == 'cell_links':
                        continue
                    clusters[key].append(value)
                clusters['rings'].append([])
                for ring_struct in aod_event['CaloRingsContainer_Rings']:
                    if ring_struct['cluster_link'] == i:
                        clusters['rings'][-1] = ring_struct['rings']
                        break
                cluster_id_counter += 1

            if batch_samples >= batch_size:
                logging.info(
                    f'Inserting batch {batch_counter} with {len(events["id"])} events')
                write_batch_to_duck_db(conn, events, clusters)
                events = defaultdict(list)
                clusters = defaultdict(list)
                batch_counter += 1
            else:
                batch_samples += 1

        # Dumps remaining data
        logging.info('Inserting remaining data')
        write_batch_to_duck_db(conn, events, clusters)
        logging.debug('Finished batch processing')


def write_batch_to_duck_db(
        conn: duckdb.DuckDBPyConnection,
        events: Dict[str, List[Any]],
        clusters: Dict[str, List[Any]]
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
    """
    events_df = pd.DataFrame.from_dict(events)  # noqa: F841 Ignores the unused variable
    conn.execute("INSERT INTO events BY NAME SELECT * FROM events_df;")
    clusters_df = pd.DataFrame.from_dict(clusters)
    for field_name in CLUSTERS_PYARROW_SCHEMA.names:
        clusters_df[field_name] = clusters_df[field_name].astype(
            pd.ArrowDtype(CLUSTERS_PYARROW_SCHEMA.field(field_name).type))
    conn.execute("INSERT INTO clusters BY NAME SELECT * FROM clusters_df;")
