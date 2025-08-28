import json
import logging
from tempfile import TemporaryDirectory
from pydantic import BaseModel, Field, PrivateAttr, computed_field
from typing import Annotated, Any, Dict, Literal
import mlflow
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import typer

from .utils import fullname


@contextmanager
def log_start_end(start_name: str = 'start',
                  end_name: str = 'end'):
    """
    Context manager to log the start and end of a function execution.
    """
    mlflow.log_param(f"{start_name}", datetime.now().isoformat())
    yield
    mlflow.log_param(f"{end_name}", datetime.now().isoformat())


ID_TYPE_HELP = "Unique identifier for the job"
IdType = Annotated[
    str | None,
    Field(
        description=ID_TYPE_HELP
    )
]

NAME_TYPE_HELP = "Name of the job"
NameType = Annotated[
    str,
    Field(
        description=NAME_TYPE_HELP
    ),
    typer.Option(
        "--name",
        help=NAME_TYPE_HELP,
    )
]

DESCRIPTION_TYPE_HELP = "Description of the job"
DescriptionType = Annotated[
    str,
    Field(
        description=DESCRIPTION_TYPE_HELP
    ),
    typer.Option(
        "--description",
        help=DESCRIPTION_TYPE_HELP,
    )
]


class MLFlowLoggedJob(BaseModel, ABC):

    id_: IdType = None
    name: NameType = 'MlFlow Logged Job'
    description: DescriptionType = ''
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
        if self.id_ is None:
            return None
        if self._mlflow_run is not None:
            return self._mlflow_run
        client = mlflow.MlflowClient()
        self._mlflow_run = client.get_run(self.id_)
        return self._mlflow_run

    @computed_field
    @property
    def run_metrics(self) -> Dict[str, float]:
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
    def from_mlflow_run(cls, run: mlflow.entities.Run) -> 'MLFlowLoggedJob':
        """
        Create an instance of the job from an MLFlow run.

        Parameters
        ----------
        run : Any
            The MLFlow run to load the job from.

        Returns
        -------
        MLFlowLoggedJob
            An instance of the job loaded from MLFlow.
        """
        instance = cls._from_mlflow_run(run)
        instance.id_ = run.info.run_id
        instance.name = run.data.tags.get(
            'mlflow.runName', cls.model_fields['name'].default)
        return instance

    @classmethod
    @abstractmethod
    def _from_mlflow_run(cls, run: Any) -> 'MLFlowLoggedJob':
        raise NotImplementedError('Subclass must implement this method')

    @classmethod
    def from_mlflow_run_id(cls, run_id: str) -> 'MLFlowLoggedJob':
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return cls.from_mlflow_run(run)

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

    def to_mlflow(self,
                  nested: bool = False,
                  tags: Dict[str, Any] = {}) -> str:
        """
        Log the job to MLFlow.

        Parameters
        ----------
        nested : bool, optional
            If True, the job will be logged as a nested run under the current active run.
            If False, it will be logged as a top-level run. Default is False.
        tags : Dict[str, Any], optional
            A dictionary of additional tags to be added to the MLFlow run.
            These tags can include any metadata relevant to the job, such as
            job type, version, or any other custom information. Default is an empty dictionary.
        Returns
        -------
        str
            The MLFlow run ID of the logged job.
        """
        with mlflow.start_run(run_name=self.name,
                              description=self.description,
                              nested=nested,
                              tags=tags) as run:
            class_name = fullname(self)
            mlflow.log_param('class_name', class_name)
            self._to_mlflow()
            return run.info.run_id

    @abstractmethod
    def _to_mlflow(self):
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):
        raise NotImplementedError("Subclasses must implement this method.")

    def execute(self,
                experiment_name: str | None = None,
                tracking_uri: str | None = None,
                nested: bool = False,
                force: Literal['all', 'error'] | None = None):

        if self.executed and force != 'all':
            logging.info('Job already executed. Skipping execution.')
            return

        start_run_params = dict(
            nested=nested,
            run_name=self.name,
            description=self.description
        )

        if self.executed and force == 'all':
            old_id = self.id_
        else:
            old_id = None
            start_run_params['run_id'] = self.id_

        with (mlflow.start_run(**start_run_params) as active_run,
              TemporaryDirectory() as tmp_dir):
            self.id_ = active_run.info.run_id
            self.exec(Path(tmp_dir),
                      experiment_name=experiment_name,
                      tracking_uri=tracking_uri,
                      force=force)
            self.executed = True
            mlflow.log_param('executed', True)

        if old_id is not None:
            client = mlflow.MlflowClient()
            client.delete_run(old_id)
            logging.info(f"Deleted old run with ID: {old_id}")

        self.__refresh_cache()
