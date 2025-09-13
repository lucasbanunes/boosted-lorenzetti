from ..dataset.duckdb import DuckDBDataset


class KANDataset(DuckDBDataset):

    def get_df_from_query(self, query: str, limit: str | None = None):
        X, y = super().get_df_from_query(query, limit)
        norms = X.sum_horizontal().abs()
        norms[norms == 0] = 1
        X[X.columns] = X/norms
        if not self.label_cols:
            y = X
        return X, y
