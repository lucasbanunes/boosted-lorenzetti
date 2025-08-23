from tempfile import TemporaryDirectory
import mlflow
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator
import mlflow.exceptions
import pandas as pd
import json
import pickle


def artifact_exists(run_id: str, artifact_path: str) -> bool:
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    return any(artifact.path == artifact_path for artifact in artifacts)


def download_without_error(run_id: str, artifact_path: str, dst_path: str) -> Path | None:
    """
    Downloads an artifact from an MLflow run without raising an error.

    Parameters
    ----------
    run_id : str
        The MLflow run ID from which to download the artifact.
    artifact_path : str
        The path to the artifact to download.
    dst_path: str
        The path to the destination directory where the artifact will be downloaded.

    Returns
    -------
    Path | None
        The path to the downloaded artifact, or None if the download failed.
    """
    try:
        return Path(mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dst_path
        ))
    except mlflow.exceptions.MlflowException as e:
        if str(e).startswith('Failed to download artifacts from path'):
            return None
        raise e


@contextmanager
def tmp_artifact_download(run_id: str,
                          artifact_path: str) -> Generator[Path | None, None, None]:
    """
    Download an artifact from a run to a temporary directory.

    Parameters
    ----------
    run_id : str
        The MLFlow run ID from which to download the artifact.
    artifact_path : str
        The path to the artifact to download.

    Yields
    ------
    Path | None
        The path to the downloaded artifact.
        None if it doesn't exist.
    """
    with TemporaryDirectory() as tmp_dir:
        yield download_without_error(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=tmp_dir
        )


def load_mlflow_csv(
    run_id: str,
    artifact_path: str,
    **kwargs
) -> pd.DataFrame:
    """
    Loads a CSV file from an MLflow run artifact.

    Parameters
    ----------
    run_id : str
        The MLflow run ID.
    artifact_path : str
        The path to the artifact in MLflow.
    **kwargs
        Additional keyword arguments to pass to `pd.read_csv`.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame.
    """
    with tmp_artifact_download(run_id, artifact_path) as tmp_path:
        return pd.read_csv(tmp_path, **kwargs)


def load_json(
        run_id: str,
        artifact_path: str
) -> Any:
    """
    Loads a JSON file from an MLflow run artifact.

    Parameters
    ----------
    run_id : str
        The MLflow run ID.
    artifact_path : str
        The path to the artifact in MLflow.

    Returns
    -------
    Any
        The loaded JSON object.
    """
    with tmp_artifact_download(run_id, artifact_path) as tmp_path:
        with open(tmp_path, 'r') as f:
            return json.load(f)


def load_pickle(
        run_id: str,
        artifact_path: str
) -> Any:
    """
    Loads a pickle file from an MLflow run artifact.

    Parameters
    ----------
    run_id : str
        The MLflow run ID.
    artifact_path : str
        The path to the artifact in MLflow.

    Returns
    -------
    Any
        The loaded pickle object.
    """
    with tmp_artifact_download(run_id, artifact_path) as tmp_path:
        with open(tmp_path, 'rb') as f:
            return pickle.load(f)
