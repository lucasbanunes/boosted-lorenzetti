import lightning as L
import duckdb
from pathlib import Path
from typing import List, Tuple
from sklearn.utils import compute_class_weight
import torch
import mlflow
import numpy as np
import pandas as pd
import polars as pl


class DuckDBDataset(L.LightningDataModule):

    def __init__(
        self,
        db_path: str | Path,
        train_query: str,
        val_query: str | None = None,
        test_query: str | None = None,
        predict_query: str | None = None,
        label_cols: str | List[str] | None = None,
        batch_size: int = 32
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

        self.save_hyperparameters()

    def __check_query(self, query: str):
        """
        Checks if the query is valid.
        """
        if not query.strip().endswith(';'):
            raise ValueError("Query must end with a semicolon.")
        if 'SELECT' not in query.upper():
            raise ValueError("Query must be a SELECT statement.")

        return query

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

    def get_dataloader_from_query(self, query: str):
        """
        Executes a query on the DuckDB database and returns a DataLoader.
        """
        X, y = self.get_df_from_query(query)
        X = X.to_torch()

        if self.label_cols:
            y = y.to_torch()
        else:
            y = X

        dataset = torch.utils.data.TensorDataset(X, y)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def __get_mlflow_dataset(self, query: str):
        X, y = self.get_df_from_query(query, limit=10)

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
        dataset = mlflow.data.from_pandas(
            df,
            source=str(self.db_path),
            name=self.db_path.stem,
            targets=','.join(y.columns.tolist())
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

    def log_to_mlflow(self):
        train_dataset = self.__get_mlflow_dataset(self.train_query)
        mlflow.log_input(train_dataset, context='training')
        if self.val_query:
            val_dataset = self.__get_mlflow_dataset(self.val_query)
            mlflow.log_input(val_dataset, context='validation')
        if self.test_query:
            test_dataset = self.__get_mlflow_dataset(self.test_query)
            mlflow.log_input(test_dataset, context='test')
        if self.predict_query:
            predict_dataset = self.__get_mlflow_dataset(self.predict_query)
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
        return self.get_dataloader_from_query(self.train_query)

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
        if self.train_query is None:
            raise ValueError("Train query is not set.")
        X, y = self.get_df_from_query(self.train_query)
        return X, y

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
        if self.val_query is None:
            raise ValueError("Validation query is not set.")
        return self.get_dataloader_from_query(self.val_query)

    def val_df(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Returns the validation DataFrames.

        Returns
        -------
        Tuple[pl.DataFrame, pl.DataFrame]
            A tuple containing the feature DataFrame and the label DataFrame for validation.
        """
        if self.val_query is None:
            raise ValueError("Validation query is not set.")
        return self.get_df_from_query(self.val_query)

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
        if self.test_query is None:
            raise ValueError("Test query is not set.")
        return self.get_dataloader_from_query(self.test_query)

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
        if self.test_query is None:
            raise ValueError("Test query is not set.")
        return self.get_df_from_query(self.test_query)

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
        if self.predict_query is None:
            raise ValueError("Prediction query is not set.")
        return self.get_dataloader_from_query(self.predict_query)

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
        if self.predict_query is None:
            raise ValueError("Prediction query is not set.")
        return self.get_df_from_query(self.predict_query)
