from functools import cached_property
import duckdb
from pathlib import Path
import polars as pl

from ..dataset.duckdb import (
    create_ringer_l1_macro,
    DuckDBDataset
)
from ..utils import unflatten_dict
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


class DuckDBDeepONetRingerDataset(DuckDBDataset):

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

        self.table_name = table_name
        self.ring_col = ring_col
        self.et_col = et_col
        self.eta_col = eta_col
        self.pileup_col = pileup_col
        self.fold_col = fold_col
        self.fold = fold
        self.label_col = label_col
        self.rings = [f'{self.ring_col}[{i}]' for i in range(1, N_RINGS+1)]

        with duckdb.connect(db_path) as conn:
            self.train_df = conn.execute(self.get_train_query()).pl()
            self.val_df = conn.execute(self.get_val_query()).pl()

        abs_eta = self.train_df[self.eta_col].abs()
        to_scale = [
            self.et_col,
            self.eta_col,
            self.pileup_col
        ]
        for
        scaler_params = {
            'mean': {
                self.train_df[self.et_col].mean(),
                abs_eta.mean(),
                self.train_df[self.pileup_col].mean(),
            },
            'std': {
                self.train_df[self.et_col].std(),
                abs_eta.std(),
                self.train_df[self.pileup_col].std(),
            }
        }

        self.train_df.with_columns([
            (pl.col(self.et_col) - .alias(self.et_col),
        ])

        with duckdb.connect(db_path) as conn:
            create_ringer_l1_macro(conn)
            params = conn.sql(STANDARD_SCALER_QUERY.format(
                et_col=self.et_col,
                eta_col=self.eta_col,
                pileup_col=self.pileup_col,
                table_name=self.table_name,
                fold_col=self.fold_col,
                fold=self.fold
            )).to_df().iloc[0].to_dict()
            self.standard_scaler_params = {}
            for key, value in params.items():
                key = key.replace('mean_', 'mean.').replace('std_', 'std.')
                self.standard_scaler_params[key] = value
            self.standard_scaler_params = unflatten_dict(self.standard_scaler_params)

        super().__init__(
            db_path=db_path,
            batch_size=batch_size,
            train_query=self.get_train_query(),
            val_query=self.get_val_query()
        )

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
            label_count = conn.execute(
                f"SELECT {self.label_col} as label, COUNT({self.label_col}) as bincount FROM {self.table_name} GROUP BY {self.label_col} HAVING {self.fold_col} != {self.fold} AND {self.fold_col} >= 0"
            ).df().sort_values(by='label')
            n_samples = label_count['bincount'].sum()
            n_classes = len(label_count)
            weights = [float(n_samples / (n_classes * count))
                       for count in label_count['bincount']]
            return weights

    def get_standard_scaled_col(self, col_name: str) -> str:
        res = f"({col_name} - {self.standard_scaler_params['mean'][col_name]}) / NULLIF({self.standard_scaler_params['std'][col_name]}, 0) as {col_name}"
        return res

    def get_train_query(self) -> str:
        rings = [f'{self.ring_col}[{i}]' for i in range(1, N_RINGS+1)]
        return f"""
        SELECT {self.et_col}, {self.eta_col}, {self.pileup_col}, {', '.join(rings)}, {self.label_col}
        FROM {self.table_name}
        WHERE {self.fold_col} != {self.fold} AND {self.fold_col} >= 0;"""

        # normalized_rings = [
        #     f'{self.ring_col}[{i}]/norm as ring_{i}' for i in range(1, N_RINGS+1)
        # ]
        # return (f"SELECT {self.get_standard_scaled_col(self.et_col)} as {self.et_col}, " +
        #         f"{self.get_standard_scaled_col(self.eta_col)} as {self.eta_col}, " +
        #         f"{self.get_standard_scaled_col(self.pileup_col)} as {self.pileup_col}, " +
        #         f"ringer_l1({self.ring_col}) as norm, " +
        #         f"{', '.join(normalized_rings)}, " +
        #         f"{self.label_col} " +
        #         f"FROM {self.table_name} WHERE {self.fold_col} != {self.fold} AND {self.fold_col} >= 0")

    def get_val_query(self) -> str:
        rings = [f'{self.ring_col}[{i}]' for i in range(1, N_RINGS+1)]
        return f"""
        SELECT {self.et_col}, {self.eta_col}, {self.pileup_col}, {', '.join(rings)}, {self.label_col}
        FROM {self.table_name}
        WHERE {self.fold_col} = {self.fold};"""
        # normalized_rings = [
        #     f'{self.ring_col}[{i}]/norm as ring_{i}' for i in range(1, N_RINGS+1)
        # ]
        # return (f"SELECT {self.get_standard_scaled_col(self.et_col)}, " +
        #         f"{self.get_standard_scaled_col(self.eta_col)}, " +
        #         f"{self.get_standard_scaled_col(self.pileup_col)}, " +
        #         f"ringer_l1({self.ring_col}) as norm, " +
        #         f"{', '.join(normalized_rings)}, " +
        #         f"{self.label_col} " +
        #         f"FROM {self.table_name} WHERE {self.fold_col} = {self.fold}")
