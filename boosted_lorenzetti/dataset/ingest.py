from pathlib import Path
from typer import Typer
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import hashlib

from . import LztDataset
from ..constants import N_RINGS
from . import ntuple


MISSING_RING = -9999
NO_FOLD_VALUE = -1
DEFAULT_DESCRIPTION = (
    'Lorenzetti dataset for electron classification.'
)


def open_rings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Opens the rings column in the DataFrame.
    """
    for i in range(N_RINGS):
        df[f'ring_{i}'] = df[ntuple.RINGS_COL].apply(lambda x: x[i] if len(x) > i else MISSING_RING)
    df.drop(columns=[ntuple.RINGS_COL], inplace=True)
    return df


def get_description(df: pd.DataFrame, no_hash: bool = False) -> dict:
    """
    Returns a description of the DataFrame.
    """
    descriptions = df.describe().to_dict()
    metrics = {}
    for field, metric_dict in descriptions.items():
        for metric, value in metric_dict.items():
            metric = metric.replace('%', '_percentage')
            metrics[f'{field}.{metric}'] = value
    metrics['size'] = len(df)
    metrics['duplicates'] = df.duplicated().sum()
    if not no_hash:
        row_hashes = pd.util.hash_pandas_object(df)
        metrics['hash'] = hashlib.sha256(row_hashes.values).hexdigest()
    return metrics


def process_dataset(dataset_path: Path,
                    label: int,
                    sample_id_start: int = 0) -> pd.DataFrame:
    """
    Processes the dataset and returns a DataFrame.
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

    description = get_description(df)
    return df, description


app = Typer()


@app.command(
    help='Prepares a lorenzetti dataset'
)
def ingest(electron_dataset: str,
           jet_dataset: str,
           name: str,
           output_dir: Path,
           lzt_version: str,
           tracking_uri: str | None = None,
           experiment_name: str = 'boosted-lorenzetti',
           n_folds: int = 5,
           seed: int = 42,
           description: str = DEFAULT_DESCRIPTION):

    if not output_dir.is_dir():
        raise ValueError(f"Output path {output_dir} is not a directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    metrics_dict = {}
    tags = {'lzt_version': lzt_version}
    electron_df, electron_description = process_dataset(electron_dataset, label=1)
    tags['electron.hash'] = electron_description.pop('hash')
    for key, value in electron_description.items():
        metrics_dict[f'electron.{key}'] = value

    jet_df, jet_description = process_dataset(jet_dataset, label=0, sample_id_start=len(electron_df))
    tags['jet.hash'] = jet_description.pop('hash')
    for key, value in jet_description.items():
        metrics_dict[f'jet.{key}'] = value

    ingest_df = pd.concat([electron_df, jet_df], ignore_index=True)
    del electron_df, jet_df
    ingest_df['fold'] = NO_FOLD_VALUE
    tags['hash'] = hashlib.sha256((tags['electron.hash']+tags['jet.hash']).encode()).hexdigest()

    ingest_description = get_description(ingest_df, no_hash=True)
    for key, value in ingest_description.items():
        metrics_dict[f'all.{key}'] = value

    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(cv.split(ingest_df, ingest_df['label'])):
        ingest_df.loc[val_idx, 'fold'] = fold

    output_path = output_dir / f'{tags["hash"]}.parquet'
    mlflow_dataset = mlflow.data.from_pandas(
        ingest_df,
        source=str(output_path),
        name=name,
        targets='label',
        digest=tags['hash']
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
        ingest_df.to_parquet(output_path, compression='gzip', index=False)

    return run.info.run_id, tags['hash']
