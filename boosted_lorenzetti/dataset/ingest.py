from pathlib import Path
from typing import Annotated, Tuple
import typer
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pyarrow as pa
import logging
from warnings import simplefilter


from . import LztDataset
from ..constants import N_RINGS
from . import ntuple
from ..log import set_logger
from .file_dataset import FileDataset
from ..data import get_dataframe_hash


MISSING_RING = -9999
NO_FOLD_VALUE = -1
DEFAULT_DESCRIPTION = (
    'Lorenzetti dataset for electron classification.'
)
# Ignores: PerformanceWarning: DataFrame is highly fragmented.
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


ElectronDatasetType = Annotated[
    str,
    typer.Argument(
        help='Path to the Lorenzetti electron dataset'
    )
]

JetDatasetType = Annotated[
    str,
    typer.Argument(
        help='Path to the Lorenzetti jet dataset'
    )
]

NameType = Annotated[
    str,
    typer.Argument(
        help='Name of the dataset to be created'
    )
]

OutputDirType = Annotated[
    Path,
    typer.Argument(
        help='Output directory where the dataset will be saved'
    )
]

LztVersionType = Annotated[
    str,
    typer.Argument(
        help='Version of the Lorenzetti dataset to be used'
    )
]

TrackingUriType = Annotated[
    str | None,
    typer.Option(
        '--tracking-uri',
        help='MLFlow tracking URI for logging the dataset job.'
    )
]

ExperimentNameType = Annotated[
    str | None,
    typer.Option(
        '--experiment-name',
        help='Name of the MLFlow experiment to log the dataset job.'
    )
]

NFoldsType = Annotated[
    int,
    typer.Option(
        '--n-folds',
        help='Number of folds for cross-validation.'
    )
]

SeedType = Annotated[
    int,
    typer.Option(
        '--seed',
        help='Random seed for reproducibility.'
    )
]

DescriptionType = Annotated[
    str,
    typer.Option(
        '--description',
        help='Description of the dataset job.'
    )
]

QueryType = Annotated[
    str | None,
    typer.Option(
        '--query',
        help='Query to filter the dataset, e.g. "abs(cl_eta) < 2.5"'
    )
]


def open_rings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands the rings column in the DataFrame into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the rings column.

    Returns
    -------
    pd.DataFrame
        The DataFrame with expanded ring columns.
    """
    for i in range(N_RINGS):
        df[f'ring_{i}'] = df[ntuple.RINGS_COL].apply(lambda x: x[i] if len(x) > i else MISSING_RING)
    df.drop(columns=[ntuple.RINGS_COL], inplace=True)
    return df


def get_description(df: pd.DataFrame, no_hash: bool = False, include_metrics: bool = False) -> dict:
    """
    Returns a dictionary with the description of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to describe.
    no_hash : bool, optional
        If True, the hash of the DataFrame will not be included in the description, by default False
    include_metrics : bool, optional
        If True, the description will include metrics like mean, std, etc. By default False

    Returns
    -------
    dict
        A dictionary with the description of the DataFrame.
    """
    metrics = {}
    if include_metrics:
        descriptions = df.describe().to_dict()
        for field, metric_dict in descriptions.items():
            for metric, value in metric_dict.items():
                metric = metric.replace('%', '_percentage')
                metrics[f'{field}.{metric}'] = value
    metrics['size'] = len(df)
    metrics['duplicates'] = df.duplicated().sum()
    if not no_hash:
        metrics['hash'] = get_dataframe_hash(df)
    return metrics


def process_dataset(dataset_path: Path,
                    label: int,
                    sample_id_start: int = 0,
                    include_metrics: bool = False,
                    query: QueryType = None) -> pd.DataFrame:
    """
    Processes a Lorenzetti dataset and returns a DataFrame with the necessary features.

    Parameters
    ----------
    dataset_path : Path
        The path to the dataset.
    label : int
        The label for the dataset.
    sample_id_start : int, optional
        The starting sample ID, by default 0
    include_metrics : bool, optional
        If True, the description will include metrics like mean, std, etc. By default False
    query : str | None, optional
        A query to filter the dataset, by default None

    Returns
    -------
    pd.DataFrame
        A DataFrame with the processed dataset.
    """
    dataset = LztDataset(dataset_path)
    df = dataset.get_ntuple_pdf()
    df = open_rings(df)
    df['id'] = np.arange(sample_id_start, sample_id_start + len(df))
    df['label'] = label
    df['mc_e'] = df['mc_e'].map(lambda x: tuple(x))
    df['mc_et'] = df['mc_et'].map(lambda x: tuple(x))
    df['mc_eta'] = df['mc_eta'].map(lambda x: tuple(x))
    df['mc_pdgid'] = df['mc_pdgid'].map(lambda x: tuple(x))
    df['mc_phi'] = df['mc_phi'].map(lambda x: tuple(x))
    df = df.convert_dtypes(dtype_backend='pyarrow')
    df['label'] = df['label'].astype(pd.ArrowDtype(pa.uint8()))
    df['id'] = df['id'].astype(pd.ArrowDtype(pa.uint64()))

    for col_name in df.columns:

        if col_name.startswith('mc_'):
            continue

        if col_name.startswith('ring_'):
            df[col_name] = df[col_name].astype(pd.ArrowDtype(pa.float32()))
            continue

        try:
            arrow_dtype = ntuple.PYARROW_SCHEMA.field(col_name).type
            df[col_name] = df[col_name].astype(pd.ArrowDtype(arrow_dtype))
        except KeyError:
            # If the column is not in the schema, keep it as is
            continue

    if query:
        df = df.query(query)

    description = get_description(df, include_metrics=include_metrics)
    return df, description


app = typer.Typer()


@app.command(
    help='Prepares a lorenzetti dataset'
)
def ingest(electron_dataset: ElectronDatasetType,
           jet_dataset: JetDatasetType,
           name: NameType,
           output_dir: OutputDirType,
           lzt_version: LztVersionType,
           tracking_uri: TrackingUriType = None,
           experiment_name: ExperimentNameType = 'boosted-lorenzetti',
           n_folds: NFoldsType = 5,
           seed: SeedType = 42,
           description: DescriptionType = DEFAULT_DESCRIPTION,
           include_metrics: bool = False,
           query: QueryType = None
           ) -> Tuple[str, str, Path]:

    if not output_dir.is_dir():
        raise ValueError(f"Output path {output_dir} is not a directory.")
    output_dir = output_dir.resolve() / name
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = FileDataset(dataset_path=output_dir)

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    set_logger()

    logging.info('Processing electron dataset...')
    metrics_dict = {}
    tags = {'lzt_version': lzt_version}
    electron_df, electron_description = process_dataset(electron_dataset,
                                                        label=1,
                                                        include_metrics=include_metrics,
                                                        query=query)
    tags['electron.hash'] = electron_description.pop('hash')
    for key, value in electron_description.items():
        metrics_dict[f'electron.{key}'] = value

    logging.info('Processing jet dataset...')
    jet_df, jet_description = process_dataset(jet_dataset,
                                              label=0,
                                              sample_id_start=len(electron_df),
                                              include_metrics=include_metrics,
                                              query=query)
    tags['jet.hash'] = jet_description.pop('hash')
    for key, value in jet_description.items():
        metrics_dict[f'jet.{key}'] = value

    logging.info('Combining datasets...')
    ingest_df = pd.concat([electron_df, jet_df], ignore_index=True)
    del electron_df, jet_df
    ingest_df['fold'] = NO_FOLD_VALUE
    tags['hash'] = get_dataframe_hash(ingest_df)

    ingest_description = get_description(ingest_df, no_hash=True)
    for key, value in ingest_description.items():
        metrics_dict[f'all.{key}'] = value

    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(cv.split(ingest_df, ingest_df['label'])):
        ingest_df.loc[val_idx, 'fold'] = fold
    ingest_df['fold'] = ingest_df['fold'].astype(pd.ArrowDtype(pa.int8()))

    output_path = output_dir / f'data_{tags["hash"]}.parquet'
    mlflow_dataset = mlflow.data.from_pandas(
        ingest_df,
        source=str(output_path.parent),
        name=name,
        targets='label',
        digest=tags['hash'],
    )
    with mlflow.start_run(run_name=f'{name} Dataset',
                          tags=tags,
                          description=description) as run:
        mlflow.log_input(
            mlflow_dataset,
            context='versioning'
        )
        mlflow.log_params(dict(
            electron_dataset=electron_dataset,
            jet_dataset=jet_dataset,
            name=name,
            output_dir=output_dir,
            lzt_version=lzt_version,
            n_folds=n_folds,
            seed=seed,
            description=description
        ))
        mlflow.log_metrics(metrics_dict)
        ds.create_df('data', ingest_df)

    logging.info(f'Ingested dataset with hash {tags["hash"]} to {output_path}')
    return run.info.run_id, tags['hash'], output_path
