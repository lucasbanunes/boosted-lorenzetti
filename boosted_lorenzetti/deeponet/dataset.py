import duckdb

from ..dataset.duckdb import (
    DuckDBDataset,
    create_standard_scaler_macros,
    create_ringer_l1_macro
)
from ..utils import unflatten_dict


class DuckDBDatasetWrapper(DuckDBDataset):
    def __init__(self,
                 db_path: str,
                 table_name: str,
                 ring_cols: str,
                 et_col: str,
                 eta_col: str,
                 pileup_col: str,
                 fold_col: str,
                 fold: int,
                 label_cols: str | list[str] | None = None,
                 batch_size: int = 32,
                 cache: bool = True):

        self.table_name = table_name
        self.ring_cols = ring_cols
        self.et_col = et_col
        self.eta_col = eta_col
        self.pileup_col = pileup_col
        self.fold_col = fold_col
        self.fold = fold

        with duckdb.connect(db_path) as conn:
            create_ringer_l1_macro(conn)
            mean_std_str = ', '.join([
                self.get_mean_std_str(self.et_col),
                self.get_mean_std_str(self.eta_col),
                self.get_mean_std_str(self.pileup_col)
            ])
            self.standard_scaler_params = conn.sql(f"SELECT {mean_std_str} FROM {self.table_name} WHERE {self.fold_col} != {self.fold} AND {self.fold_col} >= 0")
            self.standard_scaler_params = self.standard_scaler_params.to_df().to_dict()
            self.standard_scaler_params = unflatten_dict(self.standard_scaler_params)

        train_query = f"SELECT {self.et_col}, {self.eta_col}, {self.pileup_col} FROM {self.table_name} WHERE {self.fold_col} != {self.fold} AND {self.fold_col} >= 0"

    def get_mean_std_str(self, col_name: str) -> str:
        res = f"mean({col_name}) as mean.{col_name}, "
        res += f"stddev_samp({col_name}) as std.{col_name}"
        return res

    def get_standard_scaled_col(self, col_name: str) -> str:
        res = f"({col_name} - {self.standard_scaler_params['mean'][col_name]}) / NULLIF({self.standard_scaler_params['std'][col_name]}, 0) as {col_name}"
        return res

    def get_train_query(self) -> str:
        return f"SELECT {self.get_standard_scaled_col(self.et_col)}, {self.get_standard_scaled_col(self.eta_col)}, {self.get_standard_scaled_col(self.pileup_col)} FROM {self.table_name} WHERE {self.fold_col} != {self.fold} AND {self.fold_col} >= 0"
