from pydantic import BaseModel, ConfigDict
from typing import Literal, Callable, ClassVar, Tuple
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
    init: Literal['k-means++', 'stochastic-k-means++'] = 'k-means++',
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
    inertia_inits: list[np.float64] = []

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
        for _ in range(1, self.n_clusters):
            max_distance_idx = cdist(
                X, centers, metric=distance_func).min(axis=1).argmax()
            centers = np.concatenate(
                [centers, X[max_distance_idx].reshape(1, -1)], axis=0)
        return centers

    def stochastic_kmeans_plus_init(self, X: np.ndarray):
        distance_func = self.get_metric_func()
        n_samples = len(X)
        centers = X[self.generator.choice(n_samples)].reshape(1, -1)
        for _ in range(1, self.n_clusters):
            dist_closest_center = cdist(X, centers, metric=distance_func).min(axis=1)
            new_center = self.generator.choice(
                X, p=dist_closest_center/dist_closest_center.sum())
            centers = np.concatenate(
                [centers, new_center.reshape(1, -1)], axis=0)
        return centers

    def initialize_clusters(self, X: np.ndarray):
        if self.init == 'k-means++':
            return self.kmeans_plus_init(X)
        elif self.init == 'stochastic-k-means++':
            return self.stochastic_kmeans_plus_init(X)
        else:
            raise ValueError(f'Unknown init method {self.init}')

    def fit_kmeans(self, X: np.ndarray) -> Tuple[np.float64, np.ndarray]:
        distance_func = self.get_metric_func()
        centers = self.initialize_clusters(X)
        idxs = np.arange(len(X))
        previous_inertia = np.inf
        for i in range(self.max_iter):
            logging.info(f'Iteration {i}')
            distances = cdist(X, centers, metric=distance_func)
            labels = distances.argmin(axis=1)
            inertia = np.sum(distances[idxs, labels])
            if np.allclose(previous_inertia, inertia, rtol=self.tol):
                logging.info('Inertia did not change significantly. Breaking')
                break
            new_centers = np.array([X[labels == j].mean(axis=0)
                                   for j in range(self.n_clusters)])
            if np.allclose(new_centers, centers, rtol=self.tol):
                logging.info('Cluster did not change. Breaking')
                break
            previous_inertia = inertia
            centers = new_centers
        return inertia, centers

    def get_ninits_range(self) -> int:
        if self.init == 'k-means++':
            return 1
        elif self.init == 'auto':
            return 10
        else:
            return self.n_init

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        logging.info('Start fit')
        centers_inits = list()
        for i in range(self.get_ninits_range()):
            logging.info(f'Initialization {i}')
            inertia, centers = self.fit_kmeans(X)
            self.inertia_inits.append(inertia)
            centers_inits.append(centers)
        best_init = np.argmax(self.inertia_inits)
        self.cluster_centers_ = centers_inits[best_init]
        self.inertia_ = self.inertia_inits[best_init]
        logging.info('Fit end')
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
