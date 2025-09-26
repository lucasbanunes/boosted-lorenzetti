from functools import cached_property
import duckdb
from pathlib import Path
import polars as pl
import pandas as pd
import lightning as L
import torch
import mlflow

from ..dataset.duckdb import get_balanced_class_weights
from ..constants import N_RINGS


STANDARD_SCALER_QUERY = """
SELECT
    mean({et_col}) as mean_{et_col},
    stddev_samp({et_col}) as std_{et_col},
    mean(abs({eta_col})) as mean_{eta_col},
    stddev_samp(abs({eta_col})) as std_{eta_col},
    mean({pileup_col}) as mean_{pileup_col},
    stddev_samp({pileup_col}) as std_{pileup_col}
FROM {table_name}
WHERE {fold_col} != {fold} AND {fold_col} >= 0;"""


class DuckDBDeepONetRingerDataset(L.LightningDataModule):

    def __init__(self,
                 db_path: Path,
                 table_name: str,
                 ring_col: str,
                 et_col: str,
                 eta_col: str,
                 pileup_col: str,
                 fold_col: str,
                 fold: int,
                 label_col: str,
                 batch_size: int):

        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.ring_col = ring_col
        self.et_col = et_col
        self.eta_col = eta_col
        self.pileup_col = pileup_col
        self.fold_col = fold_col
        self.fold = fold
        self.label_col = label_col
        self.batch_size = batch_size

        self.rings = [f'{self.ring_col}[{i}]' for i in range(1, N_RINGS+1)]
        self.branch_input = self.rings
        self.to_scale = [
            self.et_col,
            self.eta_col,
            self.pileup_col
        ]
        self.trunk_input = self.to_scale

        self.train_query = f"""
        SELECT {self.et_col}, {self.eta_col}, {self.pileup_col}, {', '.join(self.rings)}, {self.label_col}
        FROM {self.table_name}
        WHERE {self.fold_col} != {self.fold} AND {self.fold_col} >= 0;"""

        self.val_query = f"""
        SELECT {self.et_col}, {self.eta_col}, {self.pileup_col}, {', '.join(self.rings)}, {self.label_col}
        FROM {self.table_name}
        WHERE {self.fold_col} = {self.fold} AND {self.fold_col} >= 0;"""

        with duckdb.connect(self.db_path) as conn:
            self.train_df = conn.execute(self.train_query).pl().with_columns(
                pl.col(self.eta_col).abs().alias(self.eta_col)
            )
            self.val_df = conn.execute(self.val_query).pl().with_columns(
                pl.col(self.eta_col).abs().alias(self.eta_col)
            )

        self.scaler_params = {
            'mean': {},
            'std': {}
        }
        for col_name in self.to_scale:
            col = self.train_df[col_name]
            mean = col.mean()
            std = col.std()
            self.scaler_params['mean'][col_name] = mean
            self.scaler_params['std'][col_name] = std

        self.train_df = self.preprocess_df(self.train_df)
        self.val_df = self.preprocess_df(self.val_df)

    @property
    def predict_df(self) -> pl.DataFrame:
        return pl.concat([self.train_df, self.val_df], how='vertical')

    @cached_property
    def balanced_class_weights(self) -> list[float]:
        """
        Value returned by sklearn.utils.class_weight.compute_class_weights
        with 'balanced' mode.
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html

        Returns
        -------
        list[float]
            Class weights for each class.
        """
        with duckdb.connect(self.db_path) as conn:
            return get_balanced_class_weights(
                conn,
                self.table_name,
                self.label_col,
                filter=f'{self.fold_col} != {self.fold} AND {self.fold_col} >= 0'
            )

    @cached_property
    def model_signature_df(self) -> pd.DataFrame:
        cols = self.branch_input + self.trunk_input
        return self.train_df.select(cols).head(5)

    def preprocess_df(self, df: pl.DataFrame) -> pl.DataFrame:
        for col_name in self.to_scale:
            mean = self.scaler_params['mean'][col_name]
            std = self.scaler_params['std'][col_name]
            df = df.with_columns(
                ((pl.col(col_name) - mean) / std)
                .cast(pl.Float32)
                .alias(col_name))
        ring_norms = df[self.rings].sum_horizontal().abs()
        ring_norms[ring_norms == 0] = 1
        df[self.rings] = df[self.rings] / ring_norms
        return df

    def get_dataloader(self, df: pl.DataFrame) -> torch.utils.data.DataLoader:

        dataset = torch.utils.data.TensorDataset(
            df[self.branch_input + self.trunk_input].to_torch(),
            df[self.label_col].to_torch()
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_dataloader(self):
        return self.get_dataloader(self.train_df)

    def val_dataloader(self):
        return self.get_dataloader(self.val_df)

    def predict_dataloader(self):
        return self.get_dataloader(self.predict_df)

    def __get_mlflow_dataset(self, df: pl.DataFrame, name: str | None = None):
        df = df.to_pandas(use_pyarrow_extension_array=True)
        for dtype in df.dtypes.values:
            if str(dtype) == 'object':
                df = df.convert_dtypes(dtype_backend='pyarrow')
                break
        if name is None:
            name = self.db_path.stem
        dataset = mlflow.data.from_pandas(
            df,
            source=str(self.db_path),
            name=name,
            targets=self.label_col,
        )
        return dataset

    def log_to_mlflow(self,
                      train_name: str | None = None,
                      val_name: str | None = None,
                      test_name: str | None = None,
                      predict_name: str | None = None):
        train_dataset = self.__get_mlflow_dataset(self.train_df, train_name)
        mlflow.log_input(train_dataset, context='training')
        val_dataset = self.__get_mlflow_dataset(self.val_df, val_name)
        mlflow.log_input(val_dataset, context='validation')
        predict_dataset = self.__get_mlflow_dataset(
            self.predict_df, predict_name)
        mlflow.log_input(predict_dataset, context='prediction')
