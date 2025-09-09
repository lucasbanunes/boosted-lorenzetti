from functools import cached_property
import duckdb
from pathlib import Path
import polars as pl
import lightning as L
import torch
# import pandas as pd
# import mlflow

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
        WHERE {self.fold_col} = {self.fold};"""

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

    def preprocess_df(self, df: pl.DataFrame) -> pl.DataFrame:
        for col_name in self.to_scale:
            mean = self.scaler_params['mean'][col_name]
            std = self.scaler_params['std'][col_name]
            df = df.with_columns(
                ((pl.col(col_name) - mean) / std)
                .alias(col_name))
        ring_norms = df[self.rings].sum_horizontal().abs()
        ring_norms[ring_norms == 0] = 1
        df[self.rings] = df[self.rings] / ring_norms
        return df

    def get_dataloader(self, df: pl.DataFrame) -> torch.utils.data.DataLoader:

        dataset = torch.utils.data.TensorDataset(
            df[self.branch_input].to_torch(),
            df[self.trunk_input].to_torch(),
            df[self.label_col].to_torch()
        )
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train_dataloader(self):
        return self.get_dataloader(self.train_df)

    def val_dataloader(self):
        return self.get_dataloader(self.val_df)

    # def _get_mlflow_dataset(self, query: str, name: str | None = None):
    #     X, y = self.get_df_from_query(query, limit=10)

    #     # Casting to ensure mlflow knows how to log the dataset
    #     X = X.to_pandas(use_pyarrow_extension_array=True)
    #     for dtype in X.dtypes.values:
    #         if str(dtype) == 'object':
    #             X = X.convert_dtypes(dtype_backend='pyarrow')
    #             break
    #     y = y.to_pandas(use_pyarrow_extension_array=True)
    #     for dtype in y.dtypes.values:
    #         if str(dtype) == 'object':
    #             y = y.convert_dtypes(dtype_backend='pyarrow')
    #             break

    #     df = pd.concat([X, y], axis=1)
    #     if name is None:
    #         name = self.db_path.stem
    #     dataset = mlflow.data.from_pandas(
    #         df,
    #         source=str(self.db_path),
    #         name=name,
    #         targets=','.join(y.columns.tolist()),
    #     )
    #     return dataset

    # def log_inputs_to_mlflow(self):
