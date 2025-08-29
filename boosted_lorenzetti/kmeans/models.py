from pydantic import BaseModel, ConfigDict
from typing import Literal, Callable, ClassVar
import numpy as np
from scipy.spatial.distance import cdist
import mlflow
from mlflow.pyfunc.utils import pyfunc
import logging


DistanceFuncType = Callable[[np.ndarray, np.ndarray], float]

DISTANCES_DICT = {}


class CustomMetricKMeans(BaseModel):

    SUPPORTED_SCIPY_CDIST_METRICS: ClassVar[list[str]] = [
        'braycurtis',
        'canberra',
        'chebyshev',
        'cityblock',
        'correlation',
        'cosine',
        'dice',
        'euclidean',
        'hamming',
        'jaccard',
        'jensenshannon',
        'kulczynski1',
        'mahalanobis',
        'matching',
        'minkowski',
        'rogerstanimoto',
        'russellrao',
        'seuclidean',
        'sokalmichener',
        'sokalsneath',
        'sqeuclidean',
        'yule'
    ]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_clusters: int = 8,
    init: Literal['k-means++'] = 'k-means++',
    n_init: int | Literal['auto'] = 'auto',
    max_iter: int = 300,
    tol: float = 0.0001,
    verbose: int = 0,
    random_state: int | None = None,
    copy_x: bool = True,
    algorithm: Literal['lloyd'] = 'lloyd',
    metric: Literal['euclidean'] = 'euclidean'
    generator: np.random.Generator | None = None
    cluster_centers_: np.ndarray | None = None
    inertia_: np.float64 | None = None

    def model_post_init(self, context):
        if self.generator is None:
            self.generator = np.random.default_rng(self.random_state)

    def get_metric_func(self):
        if self.metric in self.SUPPORTED_SCIPY_CDIST_METRICS:
            return self.metric
        else:
            return DISTANCES_DICT[self.metric]

    def kmeans_plus_init(self, X: np.ndarray):
        distance_func = self.get_metric_func()
        n_samples = len(X)
        centers = X[self.generator.choice(n_samples)].reshape(1, -1)
        # Randomly choose the first center
        centers[0] = X[self.generator.choice(n_samples)]
        for i in range(1, self.n_clusters):
            max_distance_idx = cdist(
                X, centers, metric=distance_func).min(axis=1).argmax()
            centers = np.concatenate(
                [centers, X[max_distance_idx].reshape(1, -1)], axis=0)
        return centers

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        logging.info('Initalizing cluster centers')
        distance_func = self.get_metric_func()
        self.cluster_centers_ = self.kmeans_plus_init(X)
        idxs = np.arange(len(X))
        previous_inertia = np.inf
        for i in range(self.max_iter):
            logging.info(f'Iteration {i}')
            distances = cdist(X, self.cluster_centers_, metric=distance_func)
            labels = distances.argmin(axis=1)
            self.inertia_ = np.sum(distances[idxs, labels])
            new_centers = np.array([X[labels == j].mean(axis=0)
                                   for j in range(self.n_clusters)])
            self.cluster_centers_ = new_centers
            if np.allclose(new_centers, self.cluster_centers_, rtol=self.tol):
                logging.info('Cluster did not change. Breaking')
                break
            elif np.allclose(previous_inertia, self.inertia_, rtol=self.tol):
                logging.info('Inertia did not change significantly. Breaking')
                break
        return self

    def predict(self, X: np.ndarray | None = None) -> np.ndarray:
        distances = cdist(X, self.cluster_centers_, metric=self.get_metric_func())
        return distances.argmin(axis=1)

    def mlflow_log_model(self, *args, **kwargs):

        @pyfunc
        def mlflow_model(model_input: list[list[float]]) -> list[int]:
            return self.model.predict(model_input).flatten().tolist()

        mlflow.pyfunc.log_model(
            python_model=mlflow_model,
            *args,
            **kwargs
        )
