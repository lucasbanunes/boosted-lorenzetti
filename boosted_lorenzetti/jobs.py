from tempfile import TemporaryDirectory
from pydantic import BaseModel, computed_field
from functools import cached_property
from typing import Any, Dict
import mlflow
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

from . import types


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
    job_name: types.JobNameType
    run_id: types.RunIdType = None
    tags: Dict[str, str] = {}

    @computed_field
    @cached_property
    def run(self) -> Any:  # Unable to use mlflow.entities.run.Run because it is not compatible with pydantic
        if self.run_id is None:
            raise ValueError("Run ID must be set before accessing the run.")
        client = mlflow.MlflowClient()
        return client.get_run(self.run_id)

    @computed_field
    @property
    def metrics(self) -> Dict[str, float]:
        return self.run.data.metrics  # type: ignore

    @abstractmethod
    @classmethod
    def from_mlflow(cls, run_id) -> 'MLFlowLoggedJob':
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
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def to_mlflow(self, nested: bool = False) -> None:
        """
        Log the job to MLFlow.

        This method should be implemented by subclasses to log the job's parameters,
        metrics, and artifacts to MLFlow.

        Parameters
        ----------
        nested : bool, optional
            If True, the job will be logged as a nested run under the current active run.
            If False, it will be logged as a top-level run. Default is False.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def custom_exec(self,
                    tmp_dir: Path,
                    experiment_name: str,
                    tracking_uri: str | None):
        raise NotImplementedError("Subclasses must implement this method.")

    def exec(self,
             experiment_name: str | None = None,
             tracking_uri: str | None = None,
             nested: bool = False):

        if self.run_id is None:
            raise ValueError(
                "Run ID must be set before executing the job. Call `to_mlflow` first.")

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

        with (mlflow.start_run(run_id=self.run_id, nested=nested,
                               run_name=self.job_name),
              TemporaryDirectory() as tmp_dir):
            with log_start_end('exec_start', 'exec_end'):
                mlflow.log_param("job", self.__class__.__name__)
                self.custom_exec(Path(tmp_dir),
                                 experiment_name=experiment_name,
                                 tracking_uri=tracking_uri)
                mlflow.log_param('completed', True)
