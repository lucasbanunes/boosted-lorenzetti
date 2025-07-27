from pathlib import Path
import pandas as pd
from typing import List


class FileDataset:
    def __init__(self, dataset_path: str | Path):
        """
        Initializes the FileDataset with the path to the dataset.

        Parameters
        ----------
        dataset_path : str | Path
            The path to the dataset directory.

        Raises
        ------
        FileNotFoundError
            The dataset path does not exist.
        ValueError
            The dataset path is not a directory.
        """
        if not isinstance(dataset_path, Path):
            dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path {dataset_path} does not exist.")

        if not dataset_path.is_dir():
            raise ValueError(
                f"Dataset path {dataset_path} is not a directory.")

        self.dataset_path = dataset_path

    def get_df_path(self, name: str) -> Path:
        """
        Constructs the full path to a DataFrame file in the dataset directory.

        Parameters
        ----------
        name : str
            The name of the parquet file.

        Returns
        -------
        Path
            The full path to the DataFrame file.
        """
        return self.dataset_path / name

    def get_df(self,
               name: str,
               load_cols: List[str] = None) -> pd.DataFrame:
        """
        Reads a DataFrame from name in the dataset directory.

        Parameters
        ----------
        name : str
            The name of the parquet file.
        load_cols : List[str], optional
            The columns to load from the parquet file, by default None

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the data.
        """
        return pd.read_parquet(
            str(self.get_df_path(name)),
            engine='pyarrow',
            columns=load_cols,
        )

    def create_df(self, name: str, df: pd.DataFrame) -> None:
        """
        Saves a DataFrame to the dataset directory.

        Parameters
        ----------
        name : str
            The name of the dataframe.
        df : pd.DataFrame
            The DataFrame to save.
        """
        df_path = self.get_df_path(name)
        df_path.mkdir(parents=True, exist_ok=True)
        df.to_parquet(
            df_path / f'{name}.parquet',
            engine='pyarrow',
            index=False,
            compression='gzip'
        )
