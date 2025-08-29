from collections import defaultdict
from functools import cached_property
from typing import Annotated, Any, ClassVar, Dict, Literal, List, Tuple
from pathlib import Path
import mlflow
import mlflow.entities
from pydantic import Field, ConfigDict, computed_field
from datetime import datetime, timezone
from mlflow.models import infer_signature
import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    accuracy_score,
    confusion_matrix
)
import json
import logging
import pickle
import pandas as pd
import plotly.express as px
import typer

from ..utils import seed_factory, unflatten_dict, flatten_dict
from ..dataset.duckdb import DuckDBDataset
from .. import mlflow as boosted_mlflow
from .. import types
from .. import jobs
from .models import CustomMetricKMeans

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


class KMeansTrainingJob(jobs.MLFlowLoggedJob):
    """
    Job for training a KMeans model.

    Attributes
    ----------
    db_path: str
        Path to the database file.
    train_query: str
        SQL query to select training data.
    val_query: str | None
        SQL query to select validation data.
    test_query: str | None
        SQL query to select test data.
    label_cols: str | None
        Name of the label column.
    n_clusters: int
        Number of clusters.
    init: str
        Method for initialization.
    n_init: int | str
        Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter: int
        Maximum number of iterations of the k-means algorithm for a single run.
    tol: float
        Relative tolerance with regards to Frobenius norm of the difference in the cluster centers.
    verbose: int
        Verbosity mode.
    random_state: int | None
        Random state for reproducibility.
    copy_x: bool
        Whether to make a copy of the input data.
    algorithm: str
        K-means algorithm to use.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    MODEL_ARTIFACT_PATH: ClassVar[str] = 'model.pkl'
    CLUSTER_CENTERS_ARTIFACT_PATH: ClassVar[str] = 'cluster_centers.json'
    INVALID_SCORE: ClassVar[float] = -1e6
    METRICS_PER_CLUSTER_ARTIFACT_PATH: ClassVar[str] = 'metrics_per_cluster.csv'

    name: jobs.NameType = 'KMeans Training Job'

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
    model: CustomMetricKMeans | None = None
    cluster_centers: Dict[str, List[float]] = {}
    metrics: Dict[str, Any] = {}
    metrics_per_cluster: pd.DataFrame | None = None

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

        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        per_cluster_evaluation = {
            'cluster': [],
            'inertia': [],
            'variance': [],
            'n_samples': [],
            'class_': []
        }
        classes, class_counts = np.unique(y_true, return_counts=True)
        class_count_dict = dict(zip(classes, class_counts))
        for class_ in class_count_dict.keys():
            per_cluster_evaluation[f'class_{class_}_samples'] = []
            per_cluster_evaluation[f'class_{class_}_ratio'] = []

        logging.info('Evaluating clustering results.')
        evaluation = {
            'inertia': inertia,
            'variance': inertia/len(X),
            # 'silhouette_score': silhouette_score(X, y_pred) if self.n_clusters > 1 else self.INVALID_SCORE,
            'calinski_harabasz_score': calinski_harabasz_score(X, y_pred) if self.n_clusters > 1 else self.INVALID_SCORE,
            # 'davies_bouldin_score': davies_bouldin_score(X, y_pred) if self.n_clusters > 1 else self.INVALID_SCORE,
            'accuracy': -1,
        }
        dict_keys = [
            'tp', 'tn', 'fp', 'fn', 'tpr', 'fpr'
        ]
        for key in dict_keys:
            evaluation[key] = {}

        logging.info('Per-cluster evaluation.')
        for i, center in enumerate(cluster_centers):
            per_cluster_evaluation['cluster'].append(i)
            in_cluster = y_pred == i
            cluster_inertia = np.sum(
                (X[in_cluster] - center.reshape(1, -1)) ** 2)
            per_cluster_evaluation['inertia'].append(cluster_inertia)
            n_samples = np.sum(in_cluster)
            per_cluster_evaluation['n_samples'].append(n_samples)
            variance = inertia / n_samples
            per_cluster_evaluation['variance'].append(variance)
            max_class_ratio = -1
            cluster_class = -1
            for class_, class_count in class_count_dict.items():
                cluster_class_samples = np.sum((y_true == class_) & in_cluster)
                class_ratio = 100*(cluster_class_samples / class_count)
                if class_ratio > max_class_ratio:
                    max_class_ratio = class_ratio
                    cluster_class = class_
                per_cluster_evaluation[f'class_{class_}_samples'].append(
                    cluster_class_samples)
                per_cluster_evaluation[f'class_{class_}_ratio'].append(
                    class_ratio)
            per_cluster_evaluation['class_'].append(cluster_class)
            del in_cluster

        per_cluster_evaluation = pd.DataFrame.from_dict(per_cluster_evaluation)

        logging.info('Classification evaluation.')
        cluster_classification = per_cluster_evaluation['class_'].to_dict()

        y_pred_classes = np.array([cluster_classification[i]
                                  for i in y_pred], dtype=np.uint8)
        evaluation['accuracy'] = accuracy_score(y_true, y_pred_classes)
        for class_ in class_count_dict.keys():
            class_y_true = (y_true == class_).astype(np.uint8)
            class_y_pred_classes = (y_pred_classes == class_).astype(np.uint8)
            cm = confusion_matrix(class_y_true, class_y_pred_classes)
            class_key = f'class_{class_}'
            evaluation['tp'][class_key] = int(cm[1, 1])
            evaluation['fp'][class_key] = int(cm[0, 1])
            evaluation['tn'][class_key] = int(cm[0, 0])
            evaluation['fn'][class_key] = int(cm[1, 0])
            evaluation['tpr'][class_key] = cm[1, 1] / \
                (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else -1
            evaluation['fpr'][class_key] = cm[0, 1] / \
                (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else -1
            del cm, class_y_true, class_y_pred_classes

        return evaluation, per_cluster_evaluation

    def log_model(self, tmp_dir):
        mlflow.sklearn.log_model(
            self.model,
            name='model',
            signature=self.model_signature
        )
        model_filepath = tmp_dir / self.MODEL_ARTIFACT_PATH
        with model_filepath.open('wb') as f:
            pickle.dump(self.model, f)
        mlflow.log_artifact(str(model_filepath))

    def log_metrics(self, tmp_dir: Path):
        mlflow.log_metrics(flatten_dict(self.metrics))
        csv_path = tmp_dir / self.METRICS_PER_CLUSTER_ARTIFACT_PATH
        self.metrics_per_cluster.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))

    def log_cluster_centers(self, tmp_dir: Path):
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
        return cluster_centers

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):
        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)
        self.datamodule.log_to_mlflow()

        self.model = CustomMetricKMeans(
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
        self.log_cluster_centers(tmp_dir)

        logging.info('Evaluating on training data')
        train_y_pred = self.model.predict(train_X)
        train_evaluation, train_per_cluster_evaluation = self.evaluate(
            train_X, train_y_pred, train_y, self.model.cluster_centers_, self.model.inertia_)
        del train_X, train_y, train_y_pred
        train_per_cluster_evaluation.rename(
            mapper=lambda col: f'train.{col}',
            axis=1,
            inplace=True
        )
        if self.metrics_per_cluster is None:
            self.metrics_per_cluster = train_per_cluster_evaluation
        else:
            self.metrics_per_cluster = pd.concat(
                [self.metrics_per_cluster, train_per_cluster_evaluation],
                axis=1
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
            val_per_cluster_evaluation.rename(
                mapper=lambda col: f'val.{col}',
                axis=1,
                inplace=True
            )
            self.metrics_per_cluster = pd.concat(
                [self.metrics_per_cluster, val_per_cluster_evaluation],
                axis=1
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
            test_per_cluster_evaluation.rename(
                mapper=lambda col: f'test.{col}',
                axis=1,
                inplace=True
            )
            self.metrics_per_cluster = pd.concat(
                [self.metrics_per_cluster, test_per_cluster_evaluation],
                axis=1
            )
            self.metrics['test'] = test_evaluation

        self.log_metrics(tmp_dir)
        self.log_model(tmp_dir)

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)


CLUSTERS_TYPE_HELP = "Number of clusters to search"
ClustersType = Annotated[
    List[int],
    Field(
        description=CLUSTERS_TYPE_HELP
    ),
]


class BestClusterNumberSearch(jobs.MLFlowLoggedJob):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    CHILDREN_METRICS_ARTIFACT_PATH: ClassVar[str] = 'search_metrics.csv'
    INERTIA_PLOT_ARTIFACT_PATH: ClassVar[str] = 'inertia_plot.html'

    name: jobs.NameType = 'Best Cluster Number Search Job'

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

        best_job_id = run.data.params.get('best_job_id', None)
        if best_job_id is not None:
            kwargs['best_job_id'] = KMeansTrainingJob.from_mlflow_run_id(
                best_job_id)
        return cls(**kwargs)

    def log_metrics(self, tmp_dir: Path):
        metrics_filepath = tmp_dir / self.CHILDREN_METRICS_ARTIFACT_PATH
        self.children_metrics.to_csv(metrics_filepath, index=False)
        mlflow.log_artifact(str(metrics_filepath))

        y_plots = ['train.inertia']
        hover_data = ['train.inertia_diff']
        labels = {'train.inertia': 'Train Inertia'}
        if 'val.inertia' in self.children_metrics.columns:
            y_plots.append('val.inertia')
            hover_data.append('val.inertia_diff')
            labels['val.inertia'] = 'Val Inertia'
        if 'test.inertia' in self.children_metrics.columns:
            y_plots.append('test.inertia')
            hover_data.append('test.inertia_diff')
            labels['test.inertia'] = 'Test Inertia'

        fig = px.line(self.children_metrics, x='n_clusters', y=y_plots,
                      markers=True, hover_data=hover_data)
        fig.update_layout(
            title='Inertia as a function of clusters',
            xaxis_title='Number of clusters',
            yaxis_title='Inertia',
            legend=dict(
                title_text='Variable'
            )
        )
        fig.for_each_trace(lambda t: t.update(name=labels[t.name]))
        fig.add_vline(x=self.best_job.n_clusters,
                      line_dash="dash",
                      line_color="black",
                      annotation_text='Best',
                      annotation_position="top left")
        mlflow.log_figure(fig, self.INERTIA_PLOT_ARTIFACT_PATH)

        mlflow.log_param('best_job_id', self.best_job.id_)
        mlflow.log_metrics(flatten_dict(self.best_job.metrics))

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
            metrics['id'].append(child.id_)
            for key, value in flatten_dict(child.metrics).items():
                metrics[key].append(value)

        self.children_metrics = pd.DataFrame.from_dict(
            metrics).sort_values(by='n_clusters').reset_index(drop=True)
        self.children_metrics['train.inertia_diff'] = self.children_metrics['train.inertia'].diff(
        )
        self.children_metrics['train.variance_diff'] = self.children_metrics['train.variance'].diff(
        )
        if 'val.inertia' in self.children_metrics.columns:
            self.children_metrics['val.inertia_diff'] = self.children_metrics['val.inertia'].diff(
            )
            self.children_metrics['val.variance_diff'] = self.children_metrics['val.variance'].diff(
            )
        if 'test.inertia' in self.children_metrics.columns:
            self.children_metrics['test.inertia_diff'] = self.children_metrics['test.inertia'].diff(
            )
            self.children_metrics['test.variance_diff'] = self.children_metrics['test.variance'].diff(
            )

        # Fills the missing fill values
        self.children_metrics.fillna(np.inf, inplace=True)

        best_job_row = self.children_metrics.loc[self.children_metrics['train.inertia_diff'].argmin(
        )]
        for child in self.children:
            if child.id_ == best_job_row['id']:
                self.best_job = child
                break
        self.log_metrics(tmp_dir)
        self.best_job.log_cluster_centers(tmp_dir)
        self.best_job.log_metrics(tmp_dir)
        self.best_job.log_model(tmp_dir)

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)


class KFoldKMeansTrainingJob(jobs.MLFlowLoggedJob):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    CHILDREN_METRICS_ARTIFACT_PATH: ClassVar[str] = 'fold_metrics.csv'

    name: jobs.NameType = 'K-Fold KMeans Training Job'

    db_path: DBPathType
    table_name: str
    feature_cols: List[str]
    best_metric: types.BestMetricType
    best_metric_mode: types.BestMetricModeType
    n_folds: int
    clusters: ClustersType
    label_col: str | None = None
    fold_col: str = 'fold'
    init: InitType = 'k-means++'
    n_init: NInitType = 'auto'
    max_iter: MaxIterType = 300
    tol: TolType = 1e-4
    verbose: VerboseType = 0
    copy_x: CopyXType = True
    algorithm: AlgorithmType = 'lloyd'
    children: List[BestClusterNumberSearch] = []
    best_search: BestClusterNumberSearch | None = None
    children_metrics: pd.DataFrame | None = None

    def get_train_query(self, fold: int) -> str:
        return f"SELECT {', '.join(self.feature_cols)}, {self.label_col} FROM {self.table_name} WHERE {self.fold_col} != {fold} AND {self.fold_col} >= 0;"

    def get_val_query(self, fold: int) -> str:
        return f"SELECT {', '.join(self.feature_cols)}, {self.label_col} FROM {self.table_name} WHERE {self.fold_col} = {fold};"

    def _to_mlflow(self):
        mlflow.log_param("db_path", self.db_path)
        mlflow.log_param("table_name", self.table_name)
        mlflow.log_param("feature_cols", json.dumps(self.feature_cols))
        mlflow.log_param("best_metric", self.best_metric)
        mlflow.log_param("best_metric_mode", self.best_metric_mode)
        mlflow.log_param("n_folds", self.n_folds)
        mlflow.log_param("clusters", json.dumps(self.clusters))
        mlflow.log_param("label_col", self.label_col)
        mlflow.log_param("fold_col", self.fold_col)
        mlflow.log_param("init", self.init)
        mlflow.log_param("n_init", self.n_init)
        mlflow.log_param("max_iter", self.max_iter)
        mlflow.log_param("tol", self.tol)
        mlflow.log_param("verbose", self.verbose)
        mlflow.log_param("copy_x", self.copy_x)
        mlflow.log_param("algorithm", self.algorithm)
        for fold in range(self.n_folds):
            logging.info(f'Creating child job for fold {fold}')
            train_query = self.get_train_query(fold)
            val_query = self.get_val_query(fold)
            child_job = BestClusterNumberSearch(
                db_path=self.db_path,
                train_query=train_query,
                val_query=val_query,
                label_cols=self.label_col,
                clusters=self.clusters,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                verbose=self.verbose,
                copy_x=self.copy_x,
                algorithm=self.algorithm
            )
            child_job.to_mlflow(nested=True,
                                tags=dict(
                                    fold=fold
                                ))
            self.children.append(child_job)

    @classmethod
    def _from_mlflow_run(cls, run: mlflow.entities.Run) -> 'KMeansTrainingJob':
        run_id = run.info.run_id
        kwargs = dict(
            db_path=run.data.params['db_path'],
            table_name=run.data.params['table_name'],
            feature_cols=json.loads(run.data.params['feature_cols']),
            best_metric=run.data.params['best_metric'],
            best_metric_mode=run.data.params['best_metric_mode'],
            n_folds=run.data.params['n_folds'],
            clusters=json.loads(run.data.params['clusters']),
            label_col=run.data.params['label_col'],
            fold_col=run.data.params['fold_col'],
            init=run.data.params['init'],
            n_init=run.data.params['n_init'],
            max_iter=run.data.params['max_iter'],
            tol=run.data.params['tol'],
            verbose=run.data.params['verbose'],
            copy_x=run.data.params['copy_x'],
            algorithm=run.data.params['algorithm'],
        )

        kwargs['children'] = [
            BestClusterNumberSearch.from_mlflow_run(run)
            for run in boosted_mlflow.get_children(run_id, [run.info.experiment_id])
        ]
        kwargs['children'] = list(
            sorted(kwargs['children'], key=lambda x: int(x.tags['fold']))
        )

        best_search_id = run.data.params.get('best_search_id', None)
        if best_search_id is not None:
            kwargs['best_search'] = BestClusterNumberSearch.from_mlflow_run_id(
                best_search_id)
        return cls(**kwargs)

    def log_metrics(self, tmp_dir: Path):
        metrics_filepath = tmp_dir / self.CHILDREN_METRICS_ARTIFACT_PATH
        self.children_metrics.to_csv(metrics_filepath, index=False)
        mlflow.log_artifact(str(metrics_filepath))
        mlflow.log_param('best_search_id', self.best_search.id_)

    def exec(self,
             tmp_dir: Path,
             experiment_name: str,
             tracking_uri: str | None,
             force: Literal['all', 'error'] | None = None):
        exec_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_start', exec_start)

        metrics_df = defaultdict(list)

        for fold, child in enumerate(self.children):
            logging.info(
                f'Fold: {fold} - Running search job {child.id_}'
            )
            child.execute(experiment_name, tracking_uri,
                          nested=True,
                          force=force)
            metrics_df['id'].append(child.id_)
            metrics_df['best_job'].append(child.best_job.id_)
            metrics_df['fold'].append(fold)
            metrics_df['n_clusters'].append(child.best_job.n_clusters)
            for key, value in flatten_dict(child.best_job.metrics).items():
                metrics_df[key].append(value)

        self.children_metrics = pd.DataFrame.from_dict(metrics_df)

        if self.best_metric_mode == 'min':
            best_search_row = self.children_metrics.loc[self.children_metrics[self.best_metric].argmin(
            )]
        else:
            best_search_row = self.children_metrics.loc[self.children_metrics[self.best_metric].argmax(
            )]

        for child in self.children:
            if child.id_ == best_search_row['id']:
                self.best_search = child
                break

        self.log_metrics(tmp_dir)
        self.best_search.log_metrics(tmp_dir)
        self.best_search.best_job.log_cluster_centers(tmp_dir)
        self.best_search.best_job.log_metrics(tmp_dir)
        self.best_search.best_job.log_model(tmp_dir)

        end_start = datetime.now(timezone.utc).timestamp()
        mlflow.log_metric('exec_end', end_start)
        mlflow.log_metric("exec_duration", end_start - exec_start)
