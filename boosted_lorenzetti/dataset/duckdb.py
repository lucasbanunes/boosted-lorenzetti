import lightning as L
import duckdb
from pathlib import Path

from ..data import tensor_dataset_from_df


class DuckDBDataset(L.LightningDataModule):

    def __init__(
        self,
        db_path: str | Path,
        train_query: str,
        val_query: str | None = None,
        test_query: str | None = None,
        predict_query: str | None = None
    ):
        self.db_path = db_path if isinstance(db_path, Path) else Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database path {self.db_path} does not exist.")
        if not self.db_path.is_file():
            raise ValueError(f"Database path {self.db_path} is not a file.")
        self.train_query = train_query
        self.val_query = val_query
        self.test_query = test_query
        self.predict_query = predict_query
    
    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.
        """
        with duckdb.connect(self.db_path) as con:
            df = con.execute(self.train_query).df()
            tensor_dataset_from_df
        con = duckdb.connect(self.db_path)
        df = con.execute(self.train_query).df()
        return L.DataLoader(df, batch_size=32, shuffle=True)