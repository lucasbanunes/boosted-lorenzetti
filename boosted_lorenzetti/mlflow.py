from tempfile import TemporaryDirectory
import mlflow
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import pandas as pd


@contextmanager
def tmp_artifact_download(run_id: str,
                          artifact_path: str) -> Generator[Path, None, None]:
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
    Generator[Path, None, None]
        The path to the downloaded artifact.
    """
    with TemporaryDirectory() as tmp_dir:
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=tmp_dir
        )
        yield Path(tmp_dir) / artifact_path


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
