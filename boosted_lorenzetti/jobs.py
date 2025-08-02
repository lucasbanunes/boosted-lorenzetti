from enum import Enum
import logging
from tempfile import TemporaryDirectory
from pydantic import BaseModel, PrivateAttr, computed_field
from typing import Annotated, Any, Dict, Literal, List
import mlflow
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

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

    BASE_DUMP_EXCLUDE = [
        'run_id', '_mlflow_run', 'run', 'tags', 'name'
    ]

    DUMP_EXCLUDE = []

    name: types.JobNameType
    run_id: types.RunIdType = None
    executed: bool = False
    _mlflow_run: Annotated[
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

    @abstractmethod
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
        return cls(**run_params['job'])

    def __refresh_cache(self):
        """
        Refresh the cached MLFlow run data.
        This method is useful to ensure that the latest data is fetched from MLFlow.
        """
        self._mlflow_run = None
        self.run

    @abstractmethod
    def to_mlflow(self,
                  nested: bool = False,
                  tags: List[str] = [],
                  extra_tags: Dict[str, Any] = {}) -> str:
        """
        Log the job to MLFlow.

        This method should be implemented by subclasses to log the job's parameters,
        metrics, and artifacts to MLFlow.

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
        params = self.model_dump(
            exclude=self.DUMP_EXCLUDE + self.BASE_DUMP_EXCLUDE
        )

        for tag in tags:
            tags[tag] = getattr(self, tag)

        with mlflow.start_run(run_name=self.name, nested=nested) as run:
            self.log_params(params)
            self.run_id = run.info.run_id
        return self.run_id

    @abstractmethod
    def custom_exec(self,
                    tmp_dir: Path,
                    experiment_name: str,
                    tracking_uri: str | None):
        raise NotImplementedError("Subclasses must implement this method.")

    def exec(self,
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
            self.run_id and ((self.run.info.status != mlflow.entities.RunStatus.FINISHED) or force)
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
            with log_start_end('exec_start', 'exec_end'):
                self.custom_exec(Path(tmp_dir),
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
