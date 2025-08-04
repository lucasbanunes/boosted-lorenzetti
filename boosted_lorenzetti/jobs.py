import json
import logging
from tempfile import TemporaryDirectory
import mlflow.artifacts
from pydantic import BaseModel, Field, PrivateAttr, computed_field
from typing import Annotated, Any, Dict, Generator, Literal, List, ClassVar
import mlflow
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import pandas as pd

from . import types
from .utils import unflatten_dict


@contextmanager
def log_start_end(start_name: str = 'start',
                  end_name: str = 'end'):
    """
    Context manager to log the start and end of a function execution.
    """
    mlflow.log_param(f"{start_name}", datetime.now().isoformat())
    yield
    mlflow.log_param(f"{end_name}", datetime.now().isoformat())


class MLFlowLoggedJob(BaseModel, ABC):

    BASE_DUMP_EXCLUDE: ClassVar[List[str]] = [
        'run_id', '_mlflow_run', 'run', 'tags', 'name', 'executed', 'metrics'
    ]

    DUMP_EXCLUDE: ClassVar[List[str]] = []

    name: str
    run_id: Annotated[
        str,
        Field(
            description="The MLFlow run ID associated with this job.",
        )
    ] = None
    executed: Annotated[
        bool,
        Field(
            description="Indicates whether the job has been executed successfully."
        )
    ] = False
    _mlflow_run: Annotated[
        Any,
        PrivateAttr()
    ] = None
    _run_dict: Annotated[
        Any,
        PrivateAttr()
    ] = None

    @computed_field
    @property
    def run(self) -> Any:  # Unable to use mlflow.entities.run.Run because it is not compatible with pydantic
        if self.run_id is None:
            return None
        if self._mlflow_run is not None:
            return self._mlflow_run
        client = mlflow.MlflowClient()
        self._mlflow_run = client.get_run(self.run_id)
        return self._mlflow_run

    @computed_field
    @property
    def metrics(self) -> Dict[str, float]:
        if self.run is None:
            return {}
        return self.run.data.metrics  # type: ignore

    @computed_field
    @property
    def tags(self) -> Dict[str, str]:
        if self.run is None:
            return {}
        return self.run.data.tags

    @computed_field
    @property
    def run_params(self) -> Dict[str, Any]:
        """
        Returns the parameters of the MLFlow run.
        """
        if self.run is None:
            return {}
        return self.run.data.params

    @computed_field
    @property
    def run_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the MLFlow run.
        """
        if self.run is None:
            self._run_dict = {}
        if self._run_dict is not None:
            return self._run_dict
        data = {
            'run_id': self.run.info.run_id,
            'name': self.run.info.run_name,
            'status': self.run.info.status,
        }
        params_dict = {
            f'param.{key}': value
            for key, value in self.run_params.items()
        }
        data.update(params_dict)

        metrics_dict = {
            f'metric.{key}': value
            for key, value in self.metrics.items()
        }
        data.update(metrics_dict)

        tags_dict = {
            f'tag.{key}': value
            for key, value in self.tags.items()
        }
        data.update(tags_dict)

        self._run_dict = data
        return self._run_dict

    @classmethod
    def load_json_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tries to load string values in the dictionary as JSON.
        If not possible, keeps the original string value.

        Parameters
        ----------
        params : Dict[str, Any]
            The parameters to load.

        Returns
        -------
        Dict[str, Any]
            A dictionary of parameters loaded from the JSON.
        """
        new_params = {}
        for key, value in params.items():
            if isinstance(value, str):
                if value == 'None':
                    new_params[key] = None
                    continue
                try:
                    new_params[key] = json.loads(value)
                except json.JSONDecodeError:
                    new_params[key] = value
            else:
                new_params[key] = value
        return new_params

    @classmethod
    def from_mlflow(cls, run_id: str) -> 'MLFlowLoggedJob':
        """
        Create an instance of the job from an MLFlow run ID.

        Parameters
        ----------
        run_id : str
            The MLFlow run ID to load the job from.

        Returns
        -------
        MLFlowLoggedJob
            An instance of the job loaded from MLFlow.
        """
        client = mlflow.MlflowClient()
        run = client.get_run(run_id)
        run_params = unflatten_dict(run.data.params)
        job_params = run_params.get('job', {})
        job_params['run_id'] = run_id
        job_params['name'] = run.info.run_name
        job_params = cls.load_json_params(job_params)
        return cls(**job_params)

    def __refresh_cache(self):
        """
        Refresh the cached MLFlow run data.
        This method is useful to ensure that the latest data is fetched from MLFlow.
        """
        self._mlflow_run = None
        self._run_dict = None
        self.run

    def convert_params(self, value: Any) -> Any:
        """
        Convert a parameter value to a format suitable for MLFlow logging.

        Parameters
        ----------
        value : Any
            The value to convert.

        Returns
        -------
        Any
            The converted value.
        """
        if isinstance(value, Path):
            return str(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, str):
            if value == 'None':
                return None
            else:
                return value
        elif value is None or isinstance(value, (int, float, bool)):
            return value
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, dict):
            return json.dumps({k: self.convert_params(v) for k, v in value.items()})
        elif isinstance(value, BaseModel):
            return self.convert_params(value.model_dump())
        elif isinstance(value, (list, tuple, set, np.ndarray)):
            return json.dumps([self.convert_params(v) for v in value])
        else:
            raise ValueError(
                f"Unsupported type for MLFlow logging: {type(value)}")

    @contextmanager
    def to_mlflow_context(self,
                          nested: bool = False,
                          tags: List[str] = [],
                          extra_tags: Dict[str, Any] = {}):
        """
        Routine to log the job to MLFlow.

        This method is used to log the job's parameters, metrics, and artifacts to MLFlow.
        It should be called within a context manager or an active MLFlow run.

        Parameters
        ----------
        nested : bool, optional
            If True, the job will be logged as a nested run under the current active run.
            If False, it will be logged as a top-level run. Default is False.
        tags : List[str], optional
            A list of class attributes to be added to the MLFlow run as tags.
        extra_tags : Dict[str, Any], optional
            A dictionary of additional tags to be added to the MLFlow run.

        Returns
        -------
        str
            The MLFlow run ID of the logged job.
        """

        params = self.model_dump(
            exclude=self.DUMP_EXCLUDE + self.BASE_DUMP_EXCLUDE
        )
        params = {
            key: self.convert_params(value)
            for key, value in params.items()
        }

        for tag in tags:
            extra_tags[tag] = getattr(self, tag)

        with mlflow.start_run(run_name=self.name, nested=nested,
                              tags=extra_tags) as run:
            self.log_params(params)
            self.run_id = run.info.run_id
            yield run

    def to_mlflow(self,
                  nested: bool = False,
                  tags: List[str] = [],
                  extra_tags: Dict[str, Any] = {}) -> str:
        """
        Log the job to MLFlow.

        Parameters
        ----------
        nested : bool, optional
            If True, the job will be logged as a nested run under the current active run.
            If False, it will be logged as a top-level run. Default is False.
        tags : List[str], optional
            A list of class attributes to be added to the MLFlow run as tags. These tags can be used
            to categorize or filter runs in the MLFlow UI. Default is an empty list.
        extra_tags : Dict[str, Any], optional
            A dictionary of additional tags to be added to the MLFlow run.
            These tags can include any metadata relevant to the job, such as
            job type, version, or any other custom information. Default is an empty dictionary.
        Returns
        -------
        str
            The MLFlow run ID of the logged job.
        """
        with self.to_mlflow_context(nested=nested,
                                    tags=tags,
                                    extra_tags=extra_tags) as run:
            return run.info.run_id

    @abstractmethod
    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None):
        raise NotImplementedError("Subclasses must implement this method.")

    def execute(self,
                experiment_name: str | None = None,
                tracking_uri: str | None = None,
                nested: bool = False,
                force: Literal['all', 'error'] | None = None):

        if self.run:
            self.executed = self.run.data.params.get('executed', False)

        if self.executed and force != 'all':
            logging.info('Job already executed. Skipping execution.')
            return

        start_run_params = dict(
            nested=nested,
            run_name=self.name,
        )

        old_id = None
        should_retry_old_run = (
            self.run_id and self.executed and (
                (self.run.info.status != mlflow.entities.RunStatus.FINISHED) or force)
        )
        if should_retry_old_run:
            old_id = self.run_id
            self.run_id = None
            self.__refresh_cache()
        elif force and not self.run_id:
            raise ValueError("Cannot force execution without a run ID.")
        else:
            start_run_params['run_id'] = self.run_id

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

        with (mlflow.start_run(run_id=self.run_id, nested=nested,
                               run_name=self.name),
              TemporaryDirectory() as tmp_dir):
            self.exec(Path(tmp_dir),
                      experiment_name=experiment_name,
                      tracking_uri=tracking_uri)
            self.executed = True
            self.log_param('executed', True)

        if old_id is not None:
            client = mlflow.MlflowClient()
            client.delete_run(old_id)
            logging.info(f"Deleted old run with ID: {old_id}")

        self.__refresh_cache()

    def log_param(self, key: str, value: Any) -> None:
        """
        Log a parameter to the MLFlow run.
        """
        mlflow.log_param(f'job.{key}', value)

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log multiple parameters to the MLFlow run.

        Parameters
        ----------
        params : Dict[str, Any]
            A dictionary of parameters to log.
        """
        for key, value in params.items():
            self.log_param(key, value)

    def artifact_exists(self, artifact_path: str) -> bool:
        """
        Check if an artifact exists in the MLFlow run.

        Parameters
        ----------
        artifact_path : str
            The path to the artifact.

        Returns
        -------
        bool
            True if the artifact exists, False otherwise.
        """
        if self.run is None:
            return False
        client = mlflow.MlflowClient()
        artifacts = client.list_artifacts(self.run_id)
        for artifact in artifacts:
            if artifact.path == artifact_path:
                return True
        return False

    @contextmanager
    def tmp_artifact_download(self,
                              run_id: str,
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

    def copy_artifact(self,
                      run_id: str,
                      artifact_path: str,
                      dst: str):
        """
        Copy an artifact from another run to the current run.

        Parameters
        ----------
        run_id : str
            The MLFlow run ID from which to copy the artifact.
        artifact_path : str
            The path to the artifact in the source run.
        dst : str
            The destination path where the artifact should be copied in the current run.
        """
        with self.tmp_artifact_download(run_id, artifact_path) as tmp_path:
            mlflow.log_artifact(str(tmp_path), dst)

    def load_csv_artifact(self, artifact_path: str,
                          run_id: str | None = None,
                          **kwargs) -> pd.DataFrame:
        """
        Load a CSV artifact from the MLFlow run.

        Parameters
        ----------
        artifact_path : str
            The path to the CSV artifact.
        run_id : str, optional
            The MLFlow run ID from which to load the artifact. If None, uses the current run ID.
        **kwargs : Any
            Additional keyword arguments to pass to `pd.read_csv`.

        Returns
        -------
        pd.DataFrame
            The loaded CSV data as a pandas DataFrame.
        """
        if run_id is None:
            run_id = self.run_id
        with self.tmp_artifact_download(run_id, artifact_path) as tmp_path:
            return pd.read_csv(tmp_path, **kwargs)
