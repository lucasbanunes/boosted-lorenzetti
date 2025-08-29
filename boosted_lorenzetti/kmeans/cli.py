from typing import Annotated, List
import mlflow
import logging
import typer

from ..utils import seed_factory
from .. import types
from . import jobs as kmeans_jobs
from .. import jobs as base_jobs


app = typer.Typer(
    name='kmeans',
    help='Utility for training KMeans models on electorn classification data.'
)


@app.command(
    help='Create a training run for a KMeans model'
)
def create_training(
    db_path: kmeans_jobs.DBPathType,
    train_query: kmeans_jobs.TrainQueryType,
    n_clusters: kmeans_jobs.NClustersType,
    val_query: kmeans_jobs.ValQueryType = kmeans_jobs.KMeansTrainingJob.model_fields['val_query'].default,
    test_query: kmeans_jobs.TestQueryType = kmeans_jobs.KMeansTrainingJob.model_fields['test_query'].default,
    label_cols: kmeans_jobs.LabelColsType = kmeans_jobs.KMeansTrainingJob.model_fields['label_cols'].default,
    init: kmeans_jobs.InitType = kmeans_jobs.KMeansTrainingJob.model_fields['init'].default,
    n_init: Annotated[
        str,
        typer.Option(
            help=kmeans_jobs.N_INIT_TYPE_HELP
        )
    ] = kmeans_jobs.KMeansTrainingJob.model_fields['n_init'].default,
    max_iter: kmeans_jobs.MaxIterType = kmeans_jobs.KMeansTrainingJob.model_fields['max_iter'].default,
    tol: kmeans_jobs.TolType = kmeans_jobs.KMeansTrainingJob.model_fields['tol'].default,
    verbose: kmeans_jobs.VerboseType = kmeans_jobs.KMeansTrainingJob.model_fields['verbose'].default,
    copy_x: kmeans_jobs.CopyXType = True,
    algorithm: kmeans_jobs.AlgorithmType = 'lloyd',
    random_state: kmeans_jobs.RandomStateType = None,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
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

    job = kmeans_jobs.KMeansTrainingJob(
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


RUN_IDS_OPTION_HELP = "List of run IDs to execute"
RunIdsOption = Annotated[
    str,
    typer.Option(
        help=RUN_IDS_OPTION_HELP)
]


@app.command(
    help='Run Kmeans Training Job'
)
def run_training(
    run_ids: RunIdsOption,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    logging.debug(
        f'Tracking URI: {tracking_uri}, Experiment Name: {experiment_name}')
    if isinstance(run_ids, str):
        run_ids = [run_id.strip() for run_id in run_ids.split(',')]

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(run_ids, str):
        run_ids = [run_ids]

    for run_id in run_ids:
        logging.info(f'Running training job with run ID: {run_id}')
        job = kmeans_jobs.KMeansTrainingJob.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)


CustersOption = Annotated[
    str,
    typer.Option(
        help=kmeans_jobs.CLUSTERS_TYPE_HELP
    )
]


@app.command(
    help='Creates a search for the best cluster number based on the max inertia diference'
)
def create_best_cluster_number_search(
    db_path: kmeans_jobs.DBPathType,
    train_query: kmeans_jobs.TrainQueryType,
    clusters: CustersOption,
    val_query: kmeans_jobs.ValQueryType = None,
    test_query: kmeans_jobs.TestQueryType = None,
    label_cols: kmeans_jobs.LabelColsType = 'label',
    init: kmeans_jobs.InitType = 'k-means++',
    n_init: Annotated[
        str,
        typer.Option(
            help=kmeans_jobs.N_INIT_TYPE_HELP
        )
    ] = 'auto',
    max_iter: kmeans_jobs.MaxIterType = 300,
    tol: kmeans_jobs.TolType = 1e-4,
    verbose: kmeans_jobs.VerboseType = 0,
    random_state: kmeans_jobs.RandomStateType = None,
    copy_x: kmeans_jobs.CopyXType = True,
    algorithm: kmeans_jobs.AlgorithmType = 'lloyd',
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    if random_state is None:
        random_state = seed_factory()

    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if isinstance(clusters, str):
        clusters = [int(c.strip()) for c in clusters.split(',')]

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    try:
        n_init = int(n_init)
    except Exception:
        pass

    job = kmeans_jobs.BestClusterNumberSearch(
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
        job = kmeans_jobs.BestClusterNumberSearch.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)


FeatureColsOption = Annotated[
    str,
    typer.Option(
        help="Comma separated list of feature columns to use for training."
    )
]


@app.command(
    help='Create KFold KMeans'
)
def create_kfold(
    db_path: kmeans_jobs.DBPathType,
    table_name: Annotated[
        str,
        typer.Option(
            help="Name of the table in the DuckDB database that contains the data."
        )
    ],
    feature_cols: FeatureColsOption,
    best_metric: types.BestMetricType,
    best_metric_mode: types.BestMetricModeType,
    n_folds: Annotated[
        int,
        typer.Option(
            help="Number of folds to use for cross-validation."
        )
    ],
    clusters: CustersOption,
    label_col: str | None = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['label_col'].default,
    fold_col: str = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['fold_col'].default,
    init: kmeans_jobs.InitType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['init'].default,
    n_init: Annotated[
        str,
        typer.Option(
            help=kmeans_jobs.N_INIT_TYPE_HELP
        )
    ] = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['n_init'].default,
    max_iter: kmeans_jobs.MaxIterType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['max_iter'].default,
    tol: kmeans_jobs.TolType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['tol'].default,
    verbose: kmeans_jobs.VerboseType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['verbose'].default,
    copy_x: kmeans_jobs.CopyXType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['copy_x'].default,
    algorithm: kmeans_jobs.AlgorithmType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['algorithm'].default,
    name: base_jobs.NameType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields['name'].default,
    description: base_jobs.DescriptionType = kmeans_jobs.KFoldKMeansTrainingJob.model_fields[
        'description'].default,
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
    logging.debug('Setting MLFlow tracking URI and experiment name.')

    if tracking_uri is None or not tracking_uri:
        tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if isinstance(clusters, str):
        clusters = [int(c.strip()) for c in clusters.split(',')]
    if isinstance(feature_cols, str):
        feature_cols = [col.strip() for col in feature_cols.split(',')]

    try:
        n_init = int(n_init)
    except Exception:
        pass

    job = kmeans_jobs.KFoldKMeansTrainingJob(
        db_path=db_path,
        table_name=table_name,
        feature_cols=feature_cols,
        best_metric=best_metric,
        best_metric_mode=best_metric_mode,
        n_folds=n_folds,
        clusters=clusters,
        label_col=label_col,
        fold_col=fold_col,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        copy_x=copy_x,
        algorithm=algorithm,
        name=name,
        description=description
    )

    run_id = job.to_mlflow()
    logging.info(f'Created training job with run ID: {run_id}')

    return run_id


@app.command(
    help='Runs KFold KMeans'
)
def run_kfold(
    run_ids: List[str],
    tracking_uri: types.TrackingUriType = None,
    experiment_name: types.ExperimentNameType = 'boosted-lorenzetti',
):
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
        job = kmeans_jobs.KFoldKMeansTrainingJob.from_mlflow_run_id(run_id)
        job.execute(experiment_name=experiment_name,
                    tracking_uri=tracking_uri)
