from functools import cached_property
from typing import Annotated, Any, ClassVar, Dict, Literal, List
from pathlib import Path
import mlflow
from sklearn.cluster import KMeans
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone
from mlflow.models import infer_signature
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    confusion_matrix
)
import json
import logging
from tempfile import TemporaryDirectory
import pickle
import typer

from ..utils import seed_factory, fullname, unflatten_dict, set_logger
from ..dataset.duckdb import DuckDBDataset
from .. import mlflow as boosted_mlflow
from .. import types

DB_PATH_TYPE_HELP = "Path to the database"
DBPathType = Annotated[
    Path,
    Field(
        description=DB_PATH_TYPE_HELP
    ),
    typer.Option(
        help=DB_PATH_TYPE_HELP
    )
]

TRAIN_QUERY_TYPE_HELP = "SQL query to select training data"
TrainQueryType = Annotated[
    str,
    Field(
        description=TRAIN_QUERY_TYPE_HELP
    ),
    typer.Option(
        help=TRAIN_QUERY_TYPE_HELP
    )
]

VAL_QUERY_TYPE_HELP = "SQL query to select validation data"
ValQueryType = Annotated[
    str | None,
    Field(
        description=VAL_QUERY_TYPE_HELP
    ),
    typer.Option(
        help=VAL_QUERY_TYPE_HELP
    )
]

TEST_QUERY_TYPE_HELP = "SQL query to select test data"
TestQueryType = Annotated[
    str | None,
    Field(
        description=TEST_QUERY_TYPE_HELP
    ),
    typer.Option(
        help=TEST_QUERY_TYPE_HELP
    )
]

LABEL_COLS_TYPE_HELP = "Name of the label column"
LabelColsType = Annotated[
    str | None,
    Field(
        description=LABEL_COLS_TYPE_HELP
    ),
    typer.Option(
        help=LABEL_COLS_TYPE_HELP
    )
]

N_CLUSTERS_TYPE_HELP = "Number of clusters"
NClustersType = Annotated[
    int,
    Field(
        description=N_CLUSTERS_TYPE_HELP
    ),
    typer.Option(
        help=N_CLUSTERS_TYPE_HELP
    )
]

INIT_TYPE_HELP = "Method for initialization"
InitType = Annotated[
    str,
    Field(
        description=INIT_TYPE_HELP
    ),
    typer.Option(
        help=INIT_TYPE_HELP
    )
]

N_INIT_TYPE_HELP = "Number of time the k-means algorithm will be run with different centroid seeds"
NInitType = Annotated[
    int | str,
    Field(
        description=N_INIT_TYPE_HELP
    ),
    typer.Option(
        help=N_INIT_TYPE_HELP
    )
]

MAX_ITER_TYPE_HELP = "Maximum number of iterations of the k-means algorithm for a single run"
MaxIterType = Annotated[
    int,
    Field(
        description=MAX_ITER_TYPE_HELP
    ),
    typer.Option(
        help=MAX_ITER_TYPE_HELP
    )
]

TOL_TYPE_HELP = "Relative tolerance with regards to Frobenius norm of the difference in the cluster centers"
TolType = Annotated[
    float,
    Field(
        description=TOL_TYPE_HELP
    ),
    typer.Option(
        help=TOL_TYPE_HELP
    )
]

VERBOSE_TYPE_HELP = "Verbosity mode"
VerboseType = Annotated[
    int,
    Field(
        description=VERBOSE_TYPE_HELP
    ),
    typer.Option(
        help=VERBOSE_TYPE_HELP
    )
]

RANDOM_STATE_TYPE_HELP = "Random state for reproducibility"
RandomStateType = Annotated[
    int | None,
    Field(
        default_factory=seed_factory,
        description=RANDOM_STATE_TYPE_HELP
    ),
    typer.Option(
        default_factory=seed_factory,
        help=RANDOM_STATE_TYPE_HELP
    )
]

COPY_X_TYPE_HELP = "Whether to make a copy of the input data"
CopyXType = Annotated[
    bool,
    Field(
        description=COPY_X_TYPE_HELP
    ),
    typer.Option(
        help=COPY_X_TYPE_HELP
    )
]

ALGORITHM_TYPE_HELP = "K-means algorithm to use"
AlgorithmType = Annotated[
    Literal['lloyd', 'elkan'],
    Field(
        default='lloyd',
        description=ALGORITHM_TYPE_HELP
    ),
    typer.Option(
        help=ALGORITHM_TYPE_HELP
    )
]

ID_TYPE_HELP = "Unique identifier for the job"
IdType = Annotated[
    str | None,
    Field(
        description=ID_TYPE_HELP
    )
]

NAME_TYPE_HELP = "Name of the training job"
NameType = Annotated[
    str,
    Field(
        description=NAME_TYPE_HELP
    )
]

DESCRIPTION_TYPE_HELP = "Description of the training job"
DescriptionType = Annotated[
    str,
    Field(
        description=DESCRIPTION_TYPE_HELP
    )
]


def parse_none_param(value: str) -> str | None:
    if value == 'None':
        return None
    return value


class KMeansTrainingJob(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_ARTIFACT_PATH: ClassVar[str] = 'model.pkl'
    CLUSTER_CENTERS_ARTIFACT_PATH: ClassVar[str] = 'cluster_centers.json'

    db_path: DBPathType
    train_query: TrainQueryType
    n_clusters: NClustersType
    val_query: ValQueryType = None
    test_query: TestQueryType = None
    label_cols: LabelColsType = 'label'
    init: InitType = 'k-means++'
    n_init: NInitType = 'auto'
    max_iter: MaxIterType = 300
    tol: TolType = 1e-4
    verbose: VerboseType = 0
    random_state: RandomStateType
    copy_x: CopyXType = True
    algorithm: AlgorithmType = 'lloyd'
    model: KMeans | None = None
    executed: bool = False
    cluster_centers: Dict[str, List[float]] = {}
    metrics: Dict[str, Any] = {}

    id_: IdType = None
    name: NameType = 'KMeans Training Job'
    description: DescriptionType = ''

    @cached_property
    def datamodule(self) -> DuckDBDataset:
        return DuckDBDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            label_cols=self.label_cols
        )

    def _to_mlflow(self):
        mlflow.log_param('db_path', self.db_path)
        mlflow.log_param('train_query', self.train_query)
        mlflow.log_param('val_query', self.val_query)
        mlflow.log_param('test_query', self.test_query)
        mlflow.log_param('label_cols', self.label_cols)
        mlflow.log_param('n_clusters', self.n_clusters)
        mlflow.log_param('init', self.init)
        mlflow.log_param('n_init', self.n_init)
        mlflow.log_param('max_iter', self.max_iter)
        mlflow.log_param('tol', self.tol)
        mlflow.log_param('verbose', self.verbose)
        mlflow.log_param('random_state', self.random_state)
        mlflow.log_param('copy_x', self.copy_x)
        mlflow.log_param('algorithm', self.algorithm)
        mlflow.log_param('executed', self.executed)

    def to_mlflow(self, nested: bool = False):
        with mlflow.start_run(run_name=self.name,
                              description=self.description,
                              nested=nested) as run:
            class_name = fullname(self)
            mlflow.log_param('class_name', class_name)
            self._to_mlflow()
            return run.info.run_id

    @classmethod
    def _from_mlflow_run(cls, run) -> 'KMeansTrainingJob':
        run_id = run.info.run_id
        kwargs = dict(
            db_path=Path(run.data.params['db_path']),
            train_query=run.data.params['train_query'],
            val_query=parse_none_param(run.data.params['val_query']),
            test_query=parse_none_param(run.data.params['test_query']),
            label_cols=run.data.params['label_cols'],
            n_clusters=int(run.data.params['n_clusters']),
            init=run.data.params['init'],
            max_iter=int(run.data.params['max_iter']),
            tol=float(run.data.params['tol']),
            verbose=int(run.data.params['verbose']),
            random_state=int(run.data.params['random_state']),
            copy_x=run.data.params['copy_x'] == 'True',
            algorithm=run.data.params['algorithm'],
            executed=run.data.params['executed'] == 'True',
        )
        if boosted_mlflow.artifact_exists(run_id, cls.MODEL_ARTIFACT_PATH):
            kwargs['model'] = boosted_mlflow.load_pickle(
                run_id=run.info.run_id,
                artifact_path=cls.MODEL_ARTIFACT_PATH
            )
        if boosted_mlflow.artifact_exists(run_id, cls.CLUSTER_CENTERS_ARTIFACT_PATH):
            kwargs['cluster_centers'] = boosted_mlflow.load_json(
                run_id=run.info.run_id,
                artifact_path=cls.CLUSTER_CENTERS_ARTIFACT_PATH
            )
        full_metrics = unflatten_dict(run.data.metrics)
        metric_keys = ['train', 'val', 'test']
        kwargs['metrics'] = {key: full_metrics.get(
            key, {}) for key in metric_keys}
        if run.data.params['n_init'] == 'auto':
            kwargs['n_init'] = 'auto'
        else:
            kwargs['n_init'] = int(run.data.params['n_init'])
        return cls(**kwargs)

    @classmethod
    def from_mlflow_run(cls, run) -> 'KMeansTrainingJob':
        instance = cls._from_mlflow_run(run)
        instance.id_ = run.info.run_id
        instance.name = run.data.tags.get(
            'mlflow.runName', cls.model_fields['name'].default)
        return instance

    @classmethod
    def from_mlflow_run_id(cls, run_id: str) -> 'KMeansTrainingJob':
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        return cls.from_mlflow_run(run)

    def evaluate(self,
                 X: np.ndarray,
                 y_pred: np.ndarray,
                 y_true: np.ndarray) -> Dict[str, Any]:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel().tolist()
        positives = tp + fn
        negatives = tn + fp
        total = positives + negatives
        evaluation = {
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'true_positive_rate': tp / positives if positives > 0 else 0,
            'false_positive_rate': fp / negatives if negatives > 0 else 0,
            'accuracy': (tp + tn) / total if total > 0 else 0,
            'samples': total,
            'silhouette_score': silhouette_score(X, y_pred),
            'calinski_harabasz_score': calinski_harabasz_score(X, y_pred),
            'davies_bouldin_score': davies_bouldin_score(X, y_pred)
        }
        return evaluation

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None):
        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        self.datamodule.log_to_mlflow()

        self.model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=self.verbose,
            random_state=self.random_state,
            copy_x=self.copy_x,
            algorithm=self.algorithm
        )

        train_X, train_y = self.datamodule.train_df()
        train_X = train_X.to_numpy()
        train_y = train_y.to_numpy()
        logging.info('Fitting Kmeans')
        self.model.fit(train_X)
        mlflow.log_metric('inertia', self.model.inertia_)
        cluster_centers = {
            i: cluster_center.tolist()
            for i, cluster_center in enumerate(
                self.model.cluster_centers_
            )
        }
        cluster_centers_filepath = tmp_dir / self.CLUSTER_CENTERS_ARTIFACT_PATH
        with open(cluster_centers_filepath, 'w') as f:
            json.dump(cluster_centers, f, indent=4)
        mlflow.log_artifact(str(cluster_centers_filepath))

        train_y_pred = self.model.predict(train_X)
        train_evaluation = self.evaluate(
            train_X, train_y_pred, train_y)
        del train_X, train_y, train_y_pred
        for k, v in train_evaluation.items():
            mlflow.log_metric(f'train.{k}', v)
        self.metrics['train'] = train_evaluation

        if self.val_query:
            val_X, val_y = self.datamodule.val_df()
            val_X = val_X.to_numpy()
            val_y = val_y.to_numpy()
            val_y_pred = self.model.predict(val_X)

            val_evaluation = self.evaluate(
                val_X, val_y_pred, val_y)
            del val_X, val_y_pred, val_y
            for k, v in val_evaluation.items():
                mlflow.log_metric(f'val.{k}', v)
            self.metrics['val'] = val_evaluation

        if self.test_query:
            test_X, test_y = self.datamodule.test_df()
            test_X = test_X.to_numpy()
            test_y = test_y.to_numpy()
            test_y_pred = self.model.predict(test_X)

            test_evaluation = self.evaluate(
                test_X, test_y_pred, test_y)
            del test_X, test_y_pred, test_y
            for k, v in test_evaluation.items():
                mlflow.log_metric(f'test.{k}', v)
            self.metrics['test'] = test_evaluation

        sample_train_X, sample_train_y = self.datamodule.get_sample()
        sample_train_X = sample_train_X.to_pandas(
            use_pyarrow_extension_array=True)
        sample_train_y = sample_train_y.to_pandas(
            use_pyarrow_extension_array=True)
        model_signature = infer_signature(sample_train_X, sample_train_y)
        mlflow.sklearn.log_model(
            self.model,
            name='model',
            signature=model_signature
        )

        model_filepath = tmp_dir / self.MODEL_ARTIFACT_PATH
        with model_filepath.open('wb') as f:
            pickle.dump(self.model, f)
        mlflow.log_artifact(str(model_filepath))

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)

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

        with (mlflow.start_run(**start_run_params) as active_run,
              TemporaryDirectory() as tmp_dir):
            self.id_ = active_run.info.run_id
            self.exec(Path(tmp_dir),
                      experiment_name=experiment_name,
                      tracking_uri=tracking_uri)
            self.executed = True
            mlflow.log_param('executed', True)

        if old_id is not None:
            client = mlflow.MlflowClient()
            client.delete_run(old_id)
            logging.info(f"Deleted old run with ID: {old_id}")


app = typer.Typer(
    name='kmeans',
    help='Utility for training KMeans models on electorn classification data.'
)


@app.command(
    help='Create a training run for a KMeans model'
)
def create_training(
    db_path: DBPathType,
    train_query: TrainQueryType,
    n_clusters: NClustersType,
    val_query: ValQueryType = KMeansTrainingJob.model_fields['val_query'].default,
    test_query: TestQueryType = KMeansTrainingJob.model_fields['test_query'].default,
    label_cols: LabelColsType = KMeansTrainingJob.model_fields['label_cols'].default,
    init: InitType = KMeansTrainingJob.model_fields['init'].default,
    n_init: NInitType = KMeansTrainingJob.model_fields['n_init'].default,
    max_iter: MaxIterType = KMeansTrainingJob.model_fields['max_iter'].default,
    tol: TolType = KMeansTrainingJob.model_fields['tol'].default,
    verbose: VerboseType = KMeansTrainingJob.model_fields['verbose'].default,
    random_state: RandomStateType = None,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    set_logger()
    if random_state is None:
        random_state = seed_factory()

    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    job = KMeansTrainingJob(
        db_path=db_path,
        train_query=train_query,
        val_query=val_query,
        test_query=test_query,
        label_cols=label_cols,
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        random_state=random_state,
    )

    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Run Kmeans Training Job'
)
def run_training(
    run_ids: List[str],
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    set_logger()
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_ids, str):
        run_ids = [run_ids]

    for run_id in run_ids:
        logging.info(f'Running training job with run ID: {run_id}')
        job = KMeansTrainingJob.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)
