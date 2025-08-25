from collections import defaultdict
from functools import cached_property
from typing import Annotated, Any, ClassVar, Dict, Literal, List, Tuple
from pathlib import Path
import mlflow
from sklearn.cluster import KMeans
from pydantic import Field, ConfigDict, computed_field
from datetime import datetime, timezone
from mlflow.models import infer_signature
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
import json
import logging
import pickle
import typer
import pandas as pd
import plotly.express as px

from ..utils import seed_factory, unflatten_dict, set_logger, flatten_dict
from ..dataset.duckdb import DuckDBDataset
from .. import mlflow as boosted_mlflow
from .. import types
from ..jobs import MLFlowLoggedJob

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
    str,
    Field(
        default='lloyd',
        description=ALGORITHM_TYPE_HELP
    ),
    typer.Option(
        help=ALGORITHM_TYPE_HELP
    )
]


def parse_none_param(value: str) -> str | None:
    if value == 'None':
        return None
    return value


class KMeansTrainingJob(MLFlowLoggedJob):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_ARTIFACT_PATH: ClassVar[str] = 'model.pkl'
    CLUSTER_CENTERS_ARTIFACT_PATH: ClassVar[str] = 'cluster_centers.json'
    INVALID_SCORE: ClassVar[float] = -1e6
    PER_CLUSTER_METRICS: ClassVar[List[str]] = [
        'inertia',
        'variance'
    ]
    METRICS_PER_CLUSTER_ARTIFACT_PATH: ClassVar[str] = 'metrics_per_cluster.csv'

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
    cluster_centers: Dict[str, List[float]] = {}
    metrics: Dict[str, Any] = {}
    metrics_per_cluster: pd.DataFrame = pd.DataFrame(
        columns=PER_CLUSTER_METRICS + ['cluster', 'data']
    )

    @cached_property
    def datamodule(self) -> DuckDBDataset:
        return DuckDBDataset(
            db_path=self.db_path,
            train_query=self.train_query,
            val_query=self.val_query,
            test_query=self.test_query,
            label_cols=self.label_cols
        )

    @computed_field
    @cached_property
    def model_signature(self) -> mlflow.models.ModelSignature:
        sample_train_X, sample_train_y = self.datamodule.get_sample()
        sample_train_X = sample_train_X.to_pandas(
            use_pyarrow_extension_array=True)
        sample_train_y = sample_train_y.to_pandas(
            use_pyarrow_extension_array=True)
        model_signature = infer_signature(sample_train_X, sample_train_y)
        return model_signature

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
            executed=run.data.params.get('executed', False)
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
        if boosted_mlflow.artifact_exists(run_id, cls.METRICS_PER_CLUSTER_ARTIFACT_PATH):
            kwargs['metrics_per_cluster'] = boosted_mlflow.load_mlflow_csv(
                run_id=run.info.run_id,
                artifact_path=cls.METRICS_PER_CLUSTER_ARTIFACT_PATH
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

    def evaluate(self,
                 X: np.ndarray,
                 y_pred: np.ndarray,
                 y_true: np.ndarray,
                 cluster_centers: np.ndarray,
                 inertia: float) -> Tuple[Dict[str, Any], pd.DataFrame]:

        per_cluster_evaluation = {key: [] for key in self.PER_CLUSTER_METRICS}
        per_cluster_evaluation['cluster'] = []
        evaluation = {}

        evaluation['silhouette_score'] = silhouette_score(X, y_pred) if self.n_clusters > 1 else self.INVALID_SCORE
        evaluation['calinski_harabasz_score'] = calinski_harabasz_score(X, y_pred) if self.n_clusters > 1 else self.INVALID_SCORE
        evaluation['davies_bouldin_score'] = davies_bouldin_score(X, y_pred) if self.n_clusters > 1 else self.INVALID_SCORE

        for i, center in enumerate(cluster_centers):
            per_cluster_evaluation['cluster'].append(i)
            in_cluster = y_pred == i
            inertia = np.sum((X[in_cluster] - center.reshape(1, -1)) ** 2)
            per_cluster_evaluation['inertia'].append(inertia)
            variance = inertia / np.sum(in_cluster)
            per_cluster_evaluation['variance'].append(variance)
            del in_cluster

        per_cluster_evaluation = pd.DataFrame.from_dict(per_cluster_evaluation)
        evaluation['inertia'] = inertia
        evaluation['variance'] = evaluation['inertia']/len(X)

        return evaluation, per_cluster_evaluation

    def log_model(self):
        mlflow.sklearn.log_model(
            self.model,
            name='model',
            signature=self.model_signature
        )

    def log_metrics(self, tmp_dir: Path):
        mlflow.log_metrics(flatten_dict(self.metrics))
        csv_path = tmp_dir / self.METRICS_PER_CLUSTER_ARTIFACT_PATH
        self.metrics_per_cluster.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):
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

        logging.info('Evaluating on training data')
        train_y_pred = self.model.predict(train_X)
        train_evaluation, train_per_cluster_evaluation = self.evaluate(
            train_X, train_y_pred, train_y, self.model.cluster_centers_, self.model.inertia_)
        del train_X, train_y, train_y_pred
        train_per_cluster_evaluation['data'] = 'train'
        self.metrics_per_cluster = pd.concat(
            [self.metrics_per_cluster, train_per_cluster_evaluation],
            axis=0
        )
        self.metrics['train'] = train_evaluation

        if self.val_query:
            logging.info('Evaluating on validation data')
            val_X, val_y = self.datamodule.val_df()
            val_X = val_X.to_numpy()
            val_y = val_y.to_numpy()
            val_y_pred = self.model.predict(val_X)

            val_evaluation, val_per_cluster_evaluation = self.evaluate(
                val_X, val_y_pred, val_y, self.model.cluster_centers_, self.model.inertia_)
            del val_X, val_y_pred, val_y
            val_per_cluster_evaluation['data'] = 'val'
            self.metrics_per_cluster = pd.concat(
                [self.metrics_per_cluster, val_per_cluster_evaluation],
                axis=0
            )
            self.metrics['val'] = val_evaluation

        if self.test_query:
            logging.info('Evaluating on test data')
            test_X, test_y = self.datamodule.test_df()
            test_X = test_X.to_numpy()
            test_y = test_y.to_numpy()
            test_y_pred = self.model.predict(test_X)

            test_evaluation, test_per_cluster_evaluation = self.evaluate(
                test_X, test_y_pred, test_y, self.model.cluster_centers_, self.model.inertia_)
            del test_X, test_y_pred, test_y
            test_per_cluster_evaluation['data'] = 'test'
            self.metrics_per_cluster = pd.concat(
                [self.metrics_per_cluster, test_per_cluster_evaluation],
                axis=0
            )
            self.metrics['test'] = test_evaluation

        self.log_metrics(tmp_dir)
        self.log_model()

        model_filepath = tmp_dir / self.MODEL_ARTIFACT_PATH
        with model_filepath.open('wb') as f:
            pickle.dump(self.model, f)
        mlflow.log_artifact(str(model_filepath))

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)


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
    n_init: Annotated[
        str,
        typer.Option(
            help=N_INIT_TYPE_HELP
        )
    ] = KMeansTrainingJob.model_fields['n_init'].default,
    max_iter: MaxIterType = KMeansTrainingJob.model_fields['max_iter'].default,
    tol: TolType = KMeansTrainingJob.model_fields['tol'].default,
    verbose: VerboseType = KMeansTrainingJob.model_fields['verbose'].default,
    copy_x: CopyXType = True,
    algorithm: AlgorithmType = 'lloyd',
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

    try:
        n_init = int(n_init)
    except Exception:
        pass

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
        copy_x=copy_x,
        algorithm=algorithm
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


CLUSTERS_TYPE_HELP = "Number of clusters to search"
ClustersType = Annotated[
    List[int],
    Field(
        default='lloyd',
        description=CLUSTERS_TYPE_HELP
    ),
    typer.Option(
        help=CLUSTERS_TYPE_HELP
    )
]


class BestClusterNumberSearch(MLFlowLoggedJob):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_ARTIFACT_PATH: ClassVar[str] = 'model.pkl'
    CLUSTER_CENTERS_ARTIFACT_PATH: ClassVar[str] = 'cluster_centers.json'
    CHILDREN_METRICS_ARTIFACT_PATH: ClassVar[str] = 'metrics.csv'
    INERTIA_PLOT_ARTIFACT_PATH: ClassVar[str] = 'inertia_plot.html'

    db_path: DBPathType
    train_query: TrainQueryType
    clusters: ClustersType
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
    cluster_centers: Dict[str, List[float]] = {}
    metrics: Dict[str, Any] = {}
    children: List[KMeansTrainingJob] = []
    children_metrics: pd.DataFrame | None = None
    best_job: KMeansTrainingJob | None = None

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
        mlflow.log_param('clusters', json.dumps(self.clusters))
        mlflow.log_param('val_query', self.val_query)
        mlflow.log_param('test_query', self.test_query)
        mlflow.log_param('label_cols', self.label_cols)
        mlflow.log_param('init', self.init)
        mlflow.log_param('n_init', self.n_init)
        mlflow.log_param('max_iter', self.max_iter)
        mlflow.log_param('tol', self.tol)
        mlflow.log_param('verbose', self.verbose)
        mlflow.log_param('random_state', self.random_state)
        mlflow.log_param('copy_x', self.copy_x)
        mlflow.log_param('algorithm', self.algorithm)

        for n_cluster in self.clusters:
            training_job = KMeansTrainingJob(
                db_path=self.db_path,
                train_query=self.train_query,
                val_query=self.val_query,
                test_query=self.test_query,
                label_cols=self.label_cols,
                n_clusters=n_cluster,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                random_state=self.random_state,
            )
            training_job.to_mlflow(nested=True)

    @classmethod
    def _from_mlflow_run(cls, run) -> 'KMeansTrainingJob':
        run_id = run.info.run_id
        kwargs = dict(
            db_path=Path(run.data.params['db_path']),
            train_query=run.data.params['train_query'],
            val_query=parse_none_param(run.data.params['val_query']),
            test_query=parse_none_param(run.data.params['test_query']),
            label_cols=run.data.params['label_cols'],
            clusters=json.loads(run.data.params['clusters']),
            init=run.data.params['init'],
            max_iter=int(run.data.params['max_iter']),
            tol=float(run.data.params['tol']),
            verbose=int(run.data.params['verbose']),
            random_state=int(run.data.params['random_state']),
            copy_x=run.data.params['copy_x'] == 'True',
            algorithm=run.data.params['algorithm'],
            executed=run.data.params.get('executed', False)
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
        if boosted_mlflow.artifact_exists(run_id, cls.CHILDREN_METRICS_ARTIFACT_PATH):
            kwargs['children_metrics'] = boosted_mlflow.load_mlflow_csv(
                run_id=run.info.run_id,
                artifact_path=cls.CHILDREN_METRICS_ARTIFACT_PATH
            )

        full_metrics = unflatten_dict(run.data.metrics)
        metric_keys = ['train', 'val', 'test']
        kwargs['metrics'] = {key: full_metrics.get(
            key, {}) for key in metric_keys}
        if run.data.params['n_init'] == 'auto':
            kwargs['n_init'] = 'auto'
        else:
            kwargs['n_init'] = int(run.data.params['n_init'])

        kwargs['children'] = [
            KMeansTrainingJob.from_mlflow_run(run)
            for run in boosted_mlflow.get_children(run_id, [run.info.experiment_id])
        ]

        best_job_id = run.data.params.get('best_job', None)
        if best_job_id is not None:
            kwargs['best_job'] = KMeansTrainingJob.from_mlflow_run_id(best_job_id)
        return cls(**kwargs)

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):
        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        self.datamodule.log_to_mlflow()

        metrics = defaultdict(list)

        for child in self.children:
            logging.info(
                f'Running child job {child.id_} with {child.n_clusters} clusters'
            )
            child.execute(experiment_name, tracking_uri,
                          nested=True,
                          force=force)
            metrics['n_clusters'].append(child.n_clusters)
            for key, value in flatten_dict(child.metrics).items():
                metrics[key].append(value)

        metrics = pd.DataFrame.from_dict(metrics).sort_values(by='n_clusters').reset_index(drop=True)
        metrics['train.inertia_diff'] = metrics['train.inertia'].diff()
        if 'val.inertia' in metrics.columns:
            metrics['val.inertia_diff'] = metrics['val.inertia'].diff()
        if 'test.inertia' in metrics.columns:
            metrics['test.inertia_diff'] = metrics['test.inertia'].diff()
        metrics_filepath = tmp_dir / self.CHILDREN_METRICS_ARTIFACT_PATH
        metrics.to_csv(metrics_filepath, index=False)
        mlflow.log_artifact(str(metrics_filepath))

        best_job_arg = metrics['train.inertia_diff'].argmin()
        self.best_job = self.children[best_job_arg]
        mlflow.log_param('best_job', self.best_job.id_)
        boosted_mlflow.copy_artifact(
            self.best_job.id_,
            self.best_job.MODEL_ARTIFACT_PATH
        )
        boosted_mlflow.copy_artifact(
            self.best_job.id_,
            self.best_job.CLUSTER_CENTERS_ARTIFACT_PATH
        )
        mlflow.log_metrics(flatten_dict(self.best_job.metrics))
        self.best_job.log_metrics(tmp_dir)
        self.best_job.log_model()

        fig = px.line(metrics, x='n_clusters', y='train.inertia')
        fig.update_layout(
            title='Inertia as a function of clusters',
            xaxis_title='Number of clusters',
            yaxis_title='Inertia'
        )
        fig.add_vline(x=self.best_job.n_clusters,
                      line_dash="dash",
                      line_color="red",
                      annotation_text=f'{self.best_job.metrics["train"]["inertia"]}',
                      annotation_position="top left")
        mlflow.log_figure(fig, self.INERTIA_PLOT_ARTIFACT_PATH)

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)


@app.command(
    help='Creates a search for the best cluster number based on the max inertia diference'
)
def create_best_cluster_number_search(
    db_path: DBPathType,
    train_query: TrainQueryType,
    clusters: ClustersType,
    val_query: ValQueryType = None,
    test_query: TestQueryType = None,
    label_cols: LabelColsType = 'label',
    init: InitType = 'k-means++',
    n_init: Annotated[
        str,
        typer.Option(
            help=N_INIT_TYPE_HELP
        )
    ] = 'auto',
    max_iter: MaxIterType = 300,
    tol: TolType = 1e-4,
    verbose: VerboseType = 0,
    random_state: RandomStateType = None,
    copy_x: CopyXType = True,
    algorithm: AlgorithmType = 'lloyd',
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

    try:
        n_init = int(n_init)
    except Exception:
        pass

    job = BestClusterNumberSearch(
        db_path=db_path,
        train_query=train_query,
        val_query=val_query,
        test_query=test_query,
        label_cols=label_cols,
        clusters=clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        random_state=random_state,
        copy_x=copy_x,
        algorithm=algorithm
    )

    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Runs a search for the best cluster number based on the max inertia difference'
)
def run_best_cluster_number_search(
    run_ids: List[str],
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    set_logger()
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')

    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_ids, str):
        run_ids = [run_ids]

    for run_id in run_ids:
        logging.info(f'Running training job with run ID: {run_id}')
        job = BestClusterNumberSearch.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)
