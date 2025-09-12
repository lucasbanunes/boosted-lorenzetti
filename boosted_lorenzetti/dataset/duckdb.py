from functools import cached_property
import logging
import lightning as L
import duckdb
from pathlib import Path
from typing import Any, List, Literal, Tuple, Annotated
from sklearn.utils import compute_class_weight
import torch
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import typer
from sklearn.model_selection import StratifiedKFold

from ..types import seed_factory


class DuckDBDataset(L.LightningDataModule):

    def __init__(
        self,
        db_path: str | Path,
        train_query: str,
        val_query: str | None = None,
        test_query: str | None = None,
        predict_query: str | None = None,
        label_cols: str | List[str] | None = None,
        batch_size: int = 32,
        cache: bool = True
    ):
        super().__init__()
        self.db_path = db_path if isinstance(db_path, Path) else Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database path {self.db_path} does not exist.")
        if not self.db_path.is_file():
            raise ValueError(f"Database path {self.db_path} is not a file.")
        self.train_query = self.__check_query(train_query)
        self.val_query = self.__check_query(val_query) if val_query else None
        self.test_query = self.__check_query(
            test_query) if test_query else None
        self.predict_query = self.__check_query(
            predict_query) if predict_query else None
        if isinstance(label_cols, str):
            self.label_cols = [label_cols]
        elif label_cols is None:
            self.label_cols = []
        else:
            self.label_cols = label_cols
        self.batch_size = batch_size
        self.cache = cache
        self.__clear_cache()

    def __clear_cache(self):
        self._train_X = None
        self._train_y = None
        self._val_X = None
        self._val_y = None
        self._test_X = None
        self._test_y = None
        self._predict_X = None
        self._predict_y = None

    def __check_query(self, query: str):
        """
        Checks if the query is valid.
        """
        if not query.strip().endswith(';'):
            raise ValueError("Query must end with a semicolon.")
        if 'SELECT' not in query.upper():
            raise ValueError("Query must be a SELECT statement.")

        return query

    @property
    def train_X(self) -> pl.DataFrame:
        """
        Returns the training features DataFrame.
        """
        if self.train_query is None:
            raise ValueError("Train query is not set.")
        elif not self.cache:
            X, _ = self.get_df_from_query(self.train_query)
            return X
        if self._train_X is None:
            self._train_X, self._train_y = self.get_df_from_query(
                self.train_query)
        return self._train_X

    @property
    def train_y(self) -> pl.DataFrame:
        """
        Returns the training labels DataFrame.
        """
        if self.train_query is None:
            raise ValueError("Train query is not set.")
        elif not self.cache:
            _, y = self.get_df_from_query(self.train_query)
            return y
        if self._train_y is None:
            self._train_X, self._train_y = self.get_df_from_query(
                self.train_query)
        return self._train_y

    @cached_property
    def feature_cols(self) -> List[str]:
        """
        Returns the feature columns from the training query.
        """
        if self.train_query is None:
            raise ValueError("Train query is not set.")
        return list(self.train_X.columns)

    @property
    def val_X(self) -> pl.DataFrame:
        """
        Returns the validation features DataFrame.
        """
        if self.val_query is None:
            raise ValueError("Validation query is not set.")
        elif not self.cache:
            X, _ = self.get_df_from_query(self.val_query)
            return X
        if self._val_X is None:
            self._val_X, self._val_y = self.get_df_from_query(self.val_query)
        return self._val_X

    @property
    def val_y(self) -> pl.DataFrame:
        """
        Returns the validation labels DataFrame.
        """
        if self.val_query is None:
            raise ValueError("Validation query is not set.")
        elif not self.cache:
            _, y = self.get_df_from_query(self.val_query)
            return y
        if self._val_y is None:
            self._val_X, self._val_y = self.get_df_from_query(self.val_query)
        return self._val_y

    @property
    def test_X(self) -> pl.DataFrame:
        """
        Returns the test features DataFrame.
        """
        if self.test_query is None:
            raise ValueError("Test query is not set.")
        elif not self.cache:
            X, _ = self.get_df_from_query(self.test_query)
            return X
        if self._test_X is None:
            self._test_X, self._test_y = self.get_df_from_query(
                self.test_query)
        return self._test_X

    @property
    def test_y(self) -> pl.DataFrame:
        """
        Returns the test labels DataFrame.
        """
        if self.test_query is None:
            raise ValueError("Test query is not set.")
        elif not self.cache:
            _, y = self.get_df_from_query(self.test_query)
            return y
        if self._test_y is None:
            self._test_X, self._test_y = self.get_df_from_query(
                self.test_query)
        return self._test_y

    @property
    def predict_X(self) -> pl.DataFrame:
        """
        Returns the prediction features DataFrame.
        """
        if self.predict_query is None:
            raise ValueError("Prediction query is not set.")
        elif not self.cache:
            X, _ = self.get_df_from_query(self.predict_query)
            return X
        if self._predict_X is None:
            self._predict_X, self._predict_y = self.get_df_from_query(
                self.predict_query)
        return self._predict_X

    @property
    def predict_y(self) -> pl.DataFrame:
        """
        Returns the prediction labels DataFrame.
        """
        if self.predict_query is None:
            raise ValueError("Prediction query is not set.")
        elif not self.cache:
            _, y = self.get_df_from_query(self.predict_query)
            return y
        if self._predict_y is None:
            self._predict_X, self._predict_y = self.get_df_from_query(
                self.predict_query)
        return self._predict_y

    def get_df_from_query(self,
                          query: str,
                          limit: int | None = None
                          ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Executes a query on the DuckDB database and returns a DataFrame.
        """
        with duckdb.connect(self.db_path) as con:
            query_exec = con.sql(query)
            if limit is not None:
                query_exec = query_exec.limit(limit)
            df = query_exec.pl()
        feature_cols = [
            col for col in df.columns if col not in self.label_cols]
        X = df[feature_cols]
        # This is done to make sure the data types are consistent
        if self.label_cols:
            y = df[self.label_cols]
        else:
            y = X
        return X, y

    def get_sample(self, limit: int = 10) -> Tuple[pl.DataFrame, pl.DataFrame]:
        return self.get_df_from_query(self.train_query, limit=limit)

    def get_dataloader(self, datatype: Literal['train', 'val', 'test', 'predict']) -> torch.utils.data.DataLoader:
        """
        Executes a query on the DuckDB database and returns a DataLoader.
        """
        X: pl.DataFrame = getattr(self, f"{datatype}_X")
        y: pl.DataFrame = getattr(self, f"{datatype}_y")

        X = X.to_torch()
        if self.label_cols:
            y = y.to_torch()
        else:
            y = X

        dataset = torch.utils.data.TensorDataset(X, y)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=(datatype == 'train'))

    def __get_mlflow_dataset(self, query: str, name: str | None = None):
        X, y = self.get_df_from_query(query)

        # Casting to ensure mlflow knows how to log the dataset
        X = X.to_pandas(use_pyarrow_extension_array=True)
        for dtype in X.dtypes.values:
            if str(dtype) == 'object':
                X = X.convert_dtypes(dtype_backend='pyarrow')
                break
        y = y.to_pandas(use_pyarrow_extension_array=True)
        for dtype in y.dtypes.values:
            if str(dtype) == 'object':
                y = y.convert_dtypes(dtype_backend='pyarrow')
                break

        df = pd.concat([X, y], axis=1)
        if name is None:
            name = self.db_path.stem
        dataset = mlflow.data.from_pandas(
            df,
            source=str(self.db_path),
            name=name,
            targets=','.join(y.columns.tolist()),
        )
        return dataset

    def get_class_weights(self,
                          how: str = 'balanced'):
        """
        Computes train class weights for the dataset.
        """
        if not self.label_cols:
            raise ValueError(
                "label_cols must be provided to compute class weights.")

        _, y = self.get_df_from_query(self.train_query)
        y = y.to_numpy().flatten()
        class_weights = compute_class_weight(
            class_weight=how,
            classes=np.unique(y),
            y=y
        )
        return class_weights

    def log_to_mlflow(self,
                      train_name: str | None = None,
                      val_name: str | None = None,
                      test_name: str | None = None,
                      predict_name: str | None = None):
        train_dataset = self.__get_mlflow_dataset(self.train_query, train_name)
        mlflow.log_input(train_dataset, context='training')
        if self.val_query:
            val_dataset = self.__get_mlflow_dataset(self.val_query, val_name)
            mlflow.log_input(val_dataset, context='validation')
        if self.test_query:
            test_dataset = self.__get_mlflow_dataset(
                self.test_query, test_name)
            mlflow.log_input(test_dataset, context='test')
        if self.predict_query:
            predict_dataset = self.__get_mlflow_dataset(
                self.predict_query, predict_name)
            mlflow.log_input(predict_dataset, context='prediction')

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the train DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            The train DataLoader.

        Raises
        ------
        ValueError
            If the train query is not set.
        """
        return self.get_dataloader('train')

    def train_df(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Returns the training DataFrames.

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the feature DataFrame and the label DataFrame for training.

        Raises
        ------
        ValueError
            If the train query is not set.
        """
        return self.train_X, self.train_y

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the validation DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation DataLoader.

        Raises
        ------
        ValueError
            If the validation query is not set.
        """
        return self.get_dataloader('val')

    def val_df(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Returns the validation DataFrames.

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the feature DataFrame and the label DataFrame for validation.
        """
        return self.val_X, self.val_y

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the test DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            The test DataLoader.

        Raises
        ------
        ValueError
            If the test query is not set.
        """
        return self.get_dataloader('test')

    def test_df(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Returns the test DataFrames.

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the feature DataFrame and the label DataFrame for testing.

        Raises
        ------
        ValueError
            If the test query is not set.
        """
        return self.test_X, self.test_y

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Returns the prediction DataLoader.

        Returns
        -------
        torch.utils.data.DataLoader
            The prediction DataLoader.

        Raises
        ------
        ValueError
            If the prediction query is not set.
        """
        return self.get_dataloader('predict')

    def predict_df(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Returns the prediction DataFrames.

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the feature DataFrame and the label DataFrame for prediction.

        Raises
        ------
        ValueError
            If the prediction query is not set.
        """
        return self.predict_X, self.predict_y


def check_table_exists(con, table_name):
    """
    Checks if a table exists in the DuckDB database.

    Args:
        con: A DuckDB connection object.
        table_name: The name of the table to check.

    Returns:
        True if the table exists, False otherwise.
    """
    query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}')"
    result = con.execute(query).fetchone()[0]
    return result


app = typer.Typer(
    name='duckdb',
    help='Duckdb database operation utilities')


@app.command(
    help='Adds a table to a DuckDB database from a list of Parquet file patterns.'
)
def add_table_from_parquet(
    files: Annotated[
        List[str],
        typer.Option(
            help='List of parquet file patterns to load. Supports the patterns supported by duckdb read_parquet function'
        )
    ],
    db_path: Annotated[
        Path,
        typer.Option(
            help='Path to the duckdb database file'
        )
    ],
    table_name: Annotated[
        str,
        typer.Option(
            help='Table name in which to save the data'
        )
    ],
    overwrite: Annotated[
        bool,
        typer.Option(
            help='Whether to overwrite the table if it exists'
        )
    ] = False
):
    """
    Adds a table to a DuckDB database from a list of Parquet file patterns.

    Parameters
    ----------
    files : List[str]
        List of parquet file patterns to load. Supports the patterns supported by duckdb read_parquet
    db_path : Path
        Path to the duckdb database file
    table_name : str
        Table name in which to save the data
    overwrite : bool
        Whether to overwrite the table if it exists
    """

    with duckdb.connect(str(db_path)) as con:
        for i, file in enumerate(files):
            logging.info(f'{i} - Adding {file} to {db_path}')
            if i < 1 and (overwrite or (not check_table_exists(con, table_name))):
                con.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{file}')")
            else:
                con.execute(
                    f"INSERT INTO {table_name} SELECT * FROM read_parquet('{file}')")


@app.command(
    help='Adds Stratified KFold column to the dataset'
)
def add_kfold(
    db_path: Annotated[
        str, typer.Option(help='Path to the DuckDB database')
    ],
    src_table: Annotated[
        str, typer.Option(help='Name of the table to add the KFold column')
    ],
    n_folds: Annotated[
        int, typer.Option(help='Number of folds for cross-validation')
    ],
    shuffle: Annotated[
        bool, typer.Option(help='Whether to shuffle the data before splitting')
    ] = True,
    random_state: Annotated[
        int | None, typer.Option(help='Random state for shuffling')
    ] = None,
    id_col: Annotated[
        str, typer.Option(help='Name of the ID column')
    ] = 'id',
    fold_col: Annotated[
        str, typer.Option(help='Name of the fold column to add')
    ] = 'fold',
    label_col: Annotated[
        str, typer.Option(help='Name of the label column')
    ] = 'label',
    filter_cond: Annotated[
        str | None, typer.Option(
            help='Optional SQL condition to filter the data before splitting')
    ] = None
):
    if random_state is None:
        random_state = seed_factory()
    with duckdb.connect(db_path) as conn:
        logging.info('Selecting data from database')
        relation = conn.table(src_table)
        if filter_cond:
            relation = relation.filter(filter_cond)
        relation = relation.select(f"{id_col}, {label_col}")
        label_df = relation.pl()
    new_col = np.full(len(label_df), -1, dtype=np.int8)
    kf = StratifiedKFold(
        n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    x_placeholder = np.full(len(label_df), 0, dtype=np.int8)
    logging.info('Creating KFold splits')
    with duckdb.connect(db_path) as conn:
        conn.execute(f"ALTER TABLE {src_table} ADD COLUMN {fold_col} TINYINT DEFAULT -1;")
    for fold, (_, val_idx) in enumerate(kf.split(x_placeholder, label_df[label_col].to_numpy())):
        new_col[val_idx] = fold
    label_df = label_df.with_columns(pl.Series(fold_col, new_col))
    del new_col, x_placeholder
    label_df.drop_in_place(label_col)
    logging.info('Writing data to database')
    with duckdb.connect(db_path) as conn:
        conn.execute(f"""
            UPDATE {src_table}
            SET {fold_col} = label_df.{fold_col}
            FROM label_df
            WHERE {src_table}.{id_col} = label_df.{id_col};
        """)


def create_standard_scaler_macros(conn: duckdb.DuckDBPyConnection):
    conn.execute("""
CREATE OR REPLACE MACRO scaling_params(table_name, column_list) AS TABLE
    FROM query_table(table_name)
    SELECT
        "avg_\\0": avg(columns(column_list)),
        "std_\\0": stddev_pop(columns(column_list));
""")
    conn.execute("""
CREATE OR REPLACE MACRO standard_scaler(val, mean, std) AS
(val - mean) / std;
""")


def create_ringer_l1_macro(conn: duckdb.DuckDBPyConnection):
    conn.execute(
        "CREATE OR REPLACE MACRO ringer_l1(val) AS abs(list_aggregate(val, 'sum'));")


CREATE_METADATA_TABLE_QUERY = """
CREATE TABLE metadata (
    name TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def add_metadata_table(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    description: str | None = None
):
    if '.' in name:
        raise ValueError("Table name cannot contain '.'")
    conn.execute(CREATE_METADATA_TABLE_QUERY)
    conn.execute("""
        INSERT INTO metadata (name, description)
        VALUES (?, ?)
    """, (name, description))


def get_metadata(
    conn: duckdb.DuckDBPyConnection,
    table_name: str = 'metadata'
) -> dict[str, Any]:
    if not check_table_exists(conn, table_name):
        raise ValueError(f"{table_name} table does not exist.")
    metadata = conn.execute(
        f"SELECT * FROM {table_name} LIMIT 1;").df().iloc[0].to_dict()
    return metadata


def get_balanced_class_weights(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    label_col: str,
    filter: str | None = None
) -> list[float]:
    if not check_table_exists(conn, table_name):
        raise ValueError(f"{table_name} table does not exist.")
    if filter is None:
        filter = ''
    else:
        filter = f'WHERE {filter}'
    label_count = conn.execute(
        f"SELECT {label_col} as label, COUNT({label_col}) as bincount FROM {table_name} {filter} GROUP BY {label_col};"
    ).df().sort_values(by='label')
    n_samples = label_count['bincount'].sum()
    n_classes = len(label_count)
    weights = [float(n_samples / (n_classes * count))
               for count in label_count['bincount']]
    return weights
