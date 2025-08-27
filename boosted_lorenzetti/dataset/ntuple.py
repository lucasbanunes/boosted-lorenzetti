from collections import defaultdict
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Any, Dict, Iterable, List, Annotated
from pathlib import Path
import ROOT
from mlflow.types import Schema, ColSpec
import duckdb
import cyclopts
import mlflow
from sklearn.model_selection import StratifiedKFold
import logging


from ..utils import open_directories, set_logger
from ..constants import N_RINGS, NO_FOLD_VALUE
from ..jobs import MLFlowLoggedJob


RINGS_COL = 'cl_rings'


SCHEMA = {
    'EventNumber': 'int32',
    'RunNumber': 'int32',
    'avgmu': 'int32',
    'cl_eta': 'float32',
    'cl_phi': 'float32',
    'cl_e': 'float32',
    'cl_et': 'float32',
    'cl_deta': 'float32',
    'cl_dphi': 'float32',
    'cl_e0': 'float32',
    'cl_e1': 'float32',
    'cl_e2': 'float32',
    'cl_e3': 'float32',
    'cl_ehad1': 'float32',
    'cl_ehad2': 'float32',
    'cl_ehad3': 'float32',
    'cl_etot': 'float32',
    'cl_e233': 'float32',
    'cl_e237': 'float32',
    'cl_e277': 'float32',
    'cl_emaxs1': 'float32',
    'cl_emaxs2': 'float32',
    'cl_e2tsts1': 'float32',
    'cl_reta': 'float32',
    'cl_rphi': 'float32',
    'cl_rhad': 'float32',
    'cl_rhad1': 'float32',
    'cl_eratio': 'float32',
    'cl_f0': 'float32',
    'cl_f1': 'float32',
    'cl_f2': 'float32',
    'cl_f3': 'float32',
    'cl_weta2': 'float32',
    'cl_rings': 'list[float32]',
    'cl_secondR': 'float32',
    'cl_lambdaCenter': 'float32',
    'cl_fracMax': 'float32',
    'cl_lateralMom': 'float32',
    'cl_longitudinalMom': 'float32',
    'el_eta': 'float32',
    'el_et': 'float32',
    'el_phi': 'float32',
    'el_tight': 'bool',
    'el_medium': 'bool',
    'el_loose': 'bool',
    'el_vloose': 'bool',
    'seed_eta': 'float32',
    'seed_phi': 'float32',
    'mc_pdgid': 'list[float32]',
    'mc_eta': 'list[float32]',
    'mc_phi': 'list[float32]',
    'mc_e': 'list[float32]',
    'mc_et': 'list[float32]',
}

SCHEMA_OPEN_VECTORS = SCHEMA.copy()
for i in range(N_RINGS):
    SCHEMA_OPEN_VECTORS[f'{RINGS_COL}.{i}'] = 'float32'
SCHEMA_OPEN_VECTORS.pop(RINGS_COL)


MLFLOW_SCHEMA = Schema([
    ColSpec(type='integer', name='EventNumber'),
    ColSpec(type='integer', name='RunNumber'),
    ColSpec(type='integer', name='avgmu'),
    ColSpec(type='float', name='cl_eta'),
    ColSpec(type='float', name='cl_phi'),
    ColSpec(type='float', name='cl_e'),
    ColSpec(type='float', name='cl_et'),
    ColSpec(type='float', name='cl_deta'),
    ColSpec(type='float', name='cl_dphi'),
    ColSpec(type='float', name='cl_e0'),
    ColSpec(type='float', name='cl_e1'),
    ColSpec(type='float', name='cl_e2'),
    ColSpec(type='float', name='cl_e3'),
    ColSpec(type='float', name='cl_ehad1'),
    ColSpec(type='float', name='cl_ehad2'),
    ColSpec(type='float', name='cl_ehad3'),
    ColSpec(type='float', name='cl_etot'),
    ColSpec(type='float', name='cl_e233'),
    ColSpec(type='float', name='cl_e237'),
    ColSpec(type='float', name='cl_e277'),
    ColSpec(type='float', name='cl_emaxs1'),
    ColSpec(type='float', name='cl_emaxs2'),
    ColSpec(type='float', name='cl_e2tsts1'),
    ColSpec(type='float', name='cl_reta'),
    ColSpec(type='float', name='cl_rphi'),
    ColSpec(type='float', name='cl_rhad'),
    ColSpec(type='float', name='cl_rhad1'),
    ColSpec(type='float', name='cl_eratio'),
    ColSpec(type='float', name='cl_f0'),
    ColSpec(type='float', name='cl_f1'),
    ColSpec(type='float', name='cl_f2'),
    ColSpec(type='float', name='cl_f3'),
    ColSpec(type='float', name='cl_weta2'),
    ColSpec(type='string', name='cl_rings'),  # List serialized as string
    ColSpec(type='float', name='cl_secondR'),
    ColSpec(type='float', name='cl_lambdaCenter'),
    ColSpec(type='float', name='cl_fracMax'),
    ColSpec(type='float', name='cl_lateralMom'),
    ColSpec(type='float', name='cl_longitudinalMom'),
    ColSpec(type='float', name='el_eta'),
    ColSpec(type='float', name='el_et'),
    ColSpec(type='float', name='el_phi'),
    ColSpec(type='boolean', name='el_tight'),
    ColSpec(type='boolean', name='el_medium'),
    ColSpec(type='boolean', name='el_loose'),
    ColSpec(type='boolean', name='el_vloose'),
    ColSpec(type='float', name='seed_eta'),
    ColSpec(type='float', name='seed_phi'),
    ColSpec(type='string', name='mc_pdgid'),  # List serialized as string
    ColSpec(type='string', name='mc_eta'),    # List serialized as string
    ColSpec(type='string', name='mc_phi'),    # List serialized as string
    ColSpec(type='string', name='mc_e'),      # List serialized as string
    ColSpec(type='string', name='mc_et')      # List serialized as string
])

PYARROW_SCHEMA = pa.schema([
    ('EventNumber', pa.int32()),
    ('RunNumber', pa.int32()),
    ('avgmu', pa.int32()),
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
    ('el_tight', pa.bool_()),
    ('el_medium', pa.bool_()),
    ('el_loose', pa.bool_()),
    ('el_vloose', pa.bool_()),
    ('seed_eta', pa.float32()),
    ('seed_phi', pa.float32()),
    ('mc_pdgid', pa.list_(pa.float32())),
    ('mc_eta', pa.list_(pa.float32())),
    ('mc_phi', pa.list_(pa.float32())),
    ('mc_e', pa.list_(pa.float32())),
    ('mc_et', pa.list_(pa.float32())),
])

FIELDS = list(PYARROW_SCHEMA.names)


def event_as_python(event,
                    open_vectors: bool = False
                    ) -> Dict[str, float]:
    """
    Convert an event object to a dictionary representation.

    Parameters
    ----------
    event : object
        The event object to convert.
    open_vectors : bool, optional
        If True, open vector fields (e.g., lists) as Python lists.

    Returns
    -------
    Dict[str, float]
        A dictionary representation of the event.
    """
    converted = {
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
    }
    if open_vectors:
        for i, ring in enumerate(event.cl_rings):
            converted[f'cl_rings.{i}'] = float(ring)
        for i, pdgid in enumerate(event.mc_pdgid):
            converted[f'mc_pdgid.{i}'] = float(pdgid)
        for i, eta in enumerate(event.mc_eta):
            converted[f'mc_eta.{i}'] = float(eta)
        for i, phi in enumerate(event.mc_phi):
            converted[f'mc_phi.{i}'] = float(phi)
        for i, e in enumerate(event.mc_e):
            converted[f'mc_e.{i}'] = float(e)
        for i, et in enumerate(event.mc_et):
            converted[f'mc_et.{i}'] = float(et)
    else:
        converted['cl_rings'] = list(event.cl_rings)
        converted['mc_pdgid'] = list(event.mc_pdgid)
        converted['mc_eta'] = list(event.mc_eta)
        converted['mc_phi'] = list(event.mc_phi)
        converted['mc_e'] = list(event.mc_e)
        converted['mc_et'] = list(event.mc_et)

    return converted


app = cyclopts.App(
    name='ntuple',
    help="Ntuple file conversion utilities",
)


def to_dict(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
            ttree_name: str = 'physics',
            open_vectors: bool = False) -> Dict[str, List[Any]]:
    """
    Convert a single ntuple root file to a dictionary representation.

    Parameters
    ----------
    input_file : str | Path
        The path to the input ntuple file.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'physics'.
    open_vectors : bool, optional
        If True, open vector fields (e.g., lists) as field_name.index columns.
        If False, keep them as lists. Default is False.

    Returns
    -------
    Dict[str, List[Any]]
        A dictionary representation of the ntuple data.
    """
    if not isinstance(input_file, ROOT.TChain):
        chain = ROOT.TChain(ttree_name)
        for file in open_directories(input_file, file_ext='root'):
            chain.Add(str(file))

    data = defaultdict(list)
    for event in chain:
        event_data = event_as_python(event, open_vectors=open_vectors)
        for col_name, value in event_data.items():
            data[col_name].append(value)
    return data


def to_pdf(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
           ttree_name: str = 'physics',
           open_vectors: bool = False) -> pd.DataFrame:
    """
    Convert a single ntuple root file to a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path
        The path to the input ntuple file.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'physics'.
    open_vectors : bool, optional
        If True, open vector fields (e.g., lists) as field_name.index columns.
        If False, keep them as lists. Default is False.
    """
    data = to_dict(input_file, ttree_name, open_vectors=open_vectors)
    data = pd.DataFrame.from_dict(data)
    for field_name in PYARROW_SCHEMA.names:
        data[field_name] = data[field_name].astype(
            pd.ArrowDtype(PYARROW_SCHEMA.field(field_name).type))
    return data


def to_pyarrow(input_file: str | Path | Iterable[Path] | Iterable[str] | ROOT.TChain,
               ttree_name: str = 'physics',
               open_vectors: bool = False) -> pa.Table:
    """
    Convert a single ntuple root file to a pandas DataFrame.

    Parameters
    ----------
    input_file : str | Path
        The path to the input ntuple file.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'physics'.
    open_vectors : bool, optional
        If True, open vector fields (e.g., lists) as field_name.index columns.
        If False, keep them as lists. Default is False.

    Returns
    -------
    pa.Table
        A PyArrow Table with the processed dataset.
    """
    data = to_dict(input_file, ttree_name, open_vectors=open_vectors)
    table = pa.Table.from_pydict(data, schema=PYARROW_SCHEMA)
    return table


@app.command(
    help='Converts ntuple files to parquet files'
)
def to_parquet(
        input_file: Annotated[
            str,
            cyclopts.Parameter(help="Path to the input ntuple file.")
        ],
        output_file: Annotated[
            str,
            cyclopts.Parameter(help="Path to the output parquet file.")
        ],
        ttree_name: Annotated[
            str,
            cyclopts.Parameter(help="NTuple Tree name inside the .root file")
        ] = 'physics',
        open_vectors: Annotated[
            bool,
            cyclopts.Parameter(
                help="If True, open vector fields (e.g., lists) as field_name.index columns.")
        ] = False) -> None:
    """
    Convert ntuple root files to a ntuple parquet file.

    Parameters
    ----------
    input_file : str
        The path to the input ntuple file.
    output_file : str
        The path where the output parquet file will be saved.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'physics'.
    open_vectors : bool, optional
        If True, open vector fields (e.g., lists) as field_name.index columns.
        If False, keep them as lists. Default is False.
    """
    df = to_pdf(input_file, ttree_name, open_vectors=open_vectors)
    df.to_parquet(output_file, index=False, compression='gzip')


@app.command()
def to_duckdb(input_file: List[str],
              output_file: str,
              ttree_name: str = 'physics',
              table_name: str = 'ntuple',
              open_vectors: bool = False) -> None:
    """
    Convert ntuple root files to a duckdb database.

    Parameters
    ----------
    input_file : List[str]
        The path to the input ntuple file.
    output_file : str
        The path where the output duckdb table will be saved.
    ttree_name : str, optional
        The name of the TTree to read from the root file. Default is 'physics'.
    table_name : str, optional
        The name of the table to create in the duckdb database. Default is 'ntuple'.
    open_vectors : bool, optional
        If True, open vector fields (e.g., lists) as field_name.index columns.
        If False, keep them as lists. Default is False.
    """
    pa_table = to_pyarrow(input_file, ttree_name, open_vectors=open_vectors)  # noqa: F841 Ignores the unused variable
    with duckdb.connect(str(output_file)) as con:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM pa_table")


def process_ntuple_path(ntuple_path: str | Path | Iterable[Path | str],
                        label: int,
                        query: str | None = None,
                        id_offset: int = 0):
    """
    Process a single ntuple path and return a DataFrame with an additional 'id' and 'label' column.

    Parameters
    ----------
    ntuple_path : str | Path | Iterable[Path | str]
        The path to the ntuple file.
    label : int
        The label to assign to the samples in the ntuple file.
    query : str | None, optional
        A query string to filter the DataFrame, by default None
    id_offset : int, optional
        An offset to apply to the 'id' column, by default 0

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the processed ntuple data.
    """
    df = to_pdf(ntuple_path)
    df['id'] = np.arange(id_offset, id_offset + len(df), dtype=np.uint32)
    df['id'] = df['id'].astype(pd.ArrowDtype(pa.uint32()))
    df['label'] = label
    df['label'] = df['label'].astype(pd.ArrowDtype(pa.uint8()))
    if query:
        df = df.query(query)

    description = {
        'size': len(df)
    }

    return df, description


class CreateDatabaseJob(MLFlowLoggedJob):
    ntuple_paths: List[Path | str] | str | Path
    labels: List[int] | int
    output_path: Path
    lzt_version: str
    n_folds: int = 5
    seed: int = 42
    description: str = "Ntuple dataset"
    table_name: str = 'ntuple'
    query: str | None = None

    def model_post_init(self, *args, **kwargs):
        super().model_post_init(*args, **kwargs)

        if isinstance(self.ntuple_paths, str):
            self.ntuple_paths = [Path(self.ntuple_paths)]
        elif isinstance(self.ntuple_paths, Path):
            self.ntuple_paths = [self.ntuple_paths]
        elif isinstance(self.ntuple_paths, Iterable) and not isinstance(self.ntuple_paths, list):
            self.ntuple_paths = [Path(p) for p in self.ntuple_paths]

        if isinstance(self.labels, int):
            self.labels = [self.labels]
        elif isinstance(self.labels, Iterable) and not isinstance(self.labels, list):
            self.labels = list(self.labels)

        if len(self.ntuple_paths) != len(self.labels):
            raise ValueError(
                "Number of ntuple paths must match number of labels.")

        if self.output_path.suffix != '.duckdb':
            self.output_path = self.output_path.with_suffix('.duckdb')

    def exec(self,
             cache_dir: Path,
             experiment_name: str,
             tracking_uri: str | None = None):

        data_df = []
        for i, label, ntuple_path in zip(range(len(self.labels)), self.labels, self.ntuple_paths):
            logging.info(f'Processing ntuple file {i}: {ntuple_path}')
            df, description = process_ntuple_path(
                ntuple_path, label, self.query, id_offset=len(data_df))
            data_df.append(df)
            for key, value in description.items():
                mlflow.log_metric(f'dataset.{i}.{key}', value)
        data_df = pd.concat(data_df, ignore_index=True)

        if self.n_folds > 0:
            logging.info('Adding folds to the dataset')
            data_df['fold'] = NO_FOLD_VALUE
            cv = StratifiedKFold(self.n_folds, shuffle=True,
                                 random_state=self.seed)
            for fold, (_, val_idx) in enumerate(cv.split(data_df, data_df['label'])):
                data_df.loc[val_idx, 'fold'] = fold
            data_df['fold'] = data_df['fold'].astype(pd.ArrowDtype(pa.uint8()))

        with duckdb.connect(str(self.output_path)) as con:
            logging.info('Writing dataset to DuckDB')
            con.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} AS SELECT * FROM data_df")

        dataset_name = self.output_path.name
        mlflow_dataset = mlflow.data.from_pandas(
            data_df,
            source=str(self.output_path),
            name=dataset_name,
            targets='label',
        )
        mlflow.log_input(
            mlflow_dataset,
            context='versioning'
        )
        logging.info('Finished')


def create_dataset(ntuple_paths: List[str],
                   labels: List[int],
                   output_path: Path,
                   lzt_version: str,
                   tracking_uri: str | None = None,
                   experiment_name: str = 'boosted-lorenzetti',
                   n_folds: int = CreateDatabaseJob.model_fields['n_folds'].default,
                   seed: int = CreateDatabaseJob.model_fields['seed'].default,
                   description: str = CreateDatabaseJob.model_fields['description'].default,
                   table_name: str = CreateDatabaseJob.model_fields['table_name'].default,
                   query: str | None = CreateDatabaseJob.model_fields['query'].default) -> Path:
    """
    Create a dataset from the ntuple root files.
    """

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    set_logger()

    name = output_path.name.replace(
        '.duckdb', '').replace('_', ' ').capitalize()
    job = CreateDatabaseJob(
        name=f'{name} Dataset',
        ntuple_paths=ntuple_paths,
        labels=labels,
        output_path=output_path,
        lzt_version=lzt_version,
        n_folds=n_folds,
        seed=seed,
        description=description,
        table_name=table_name,
        query=query
    )
    job.to_mlflow(tags=['lzt_version'])
    job.execute()

    return job.output_path


# Typer doesn seem to work well with multiple list arguments.
# we get an error TypeError: Cannot have two nargs < 0
# https://github.com/fastapi/typer/issues/260
@app.command(
    help="Create a dataset from the ntuple root files."
)
def cli_create_dataset(
    ntuple_files: List[str],
    output_path: Path = cyclopts.Parameter(...,
                                           help="Path to the output DuckDB file."),
    lzt_version: str = cyclopts.Parameter(...,
                                          help="Version of the boosted-lorenzetti library."),
    tracking_uri: str | None = None,
    experiment_name: str = 'boosted-lorenzetti',
    n_folds: int = CreateDatabaseJob.model_fields['n_folds'].default,
    seed: int = CreateDatabaseJob.model_fields['seed'].default,
    description: str = CreateDatabaseJob.model_fields['description'].default,
    table_name: str = CreateDatabaseJob.model_fields['table_name'].default,
    query: str | None = CreateDatabaseJob.model_fields['query'].default
):
    ntuple_paths = [Path(p) for p in ntuple_files[::2]]
    labels = [int(value) for value in ntuple_files[1::2]]
    return create_dataset(
        ntuple_paths=ntuple_paths,
        labels=labels,
        output_path=output_path,
        lzt_version=lzt_version,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        n_folds=n_folds,
        seed=seed,
        description=description,
        table_name=table_name,
        query=query
    )
