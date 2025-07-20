from typing import Generator, Tuple
import pandas as pd
import numpy as np


class ColumnKFold:
    def __init__(self,
                 fold_col: str = 'fold'):
        self.fold_col = fold_col

    def get_fold_idx(self, data: pd.DataFrame,
                     fold: int) -> Tuple[pd.Series, pd.Series]:
        """
        Get the indices of the training and validation data for a specific fold.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing features and labels.
        fold : int
            The fold number to retrieve the indices for.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training indices and validation indices for the specified fold.
        """
        train_idx = data[self.fold_col] != fold
        val_idx = data[self.fold_col] == fold
        return train_idx, val_idx

    def split(self, data: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate training and validation indices for each fold.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing features and labels.

        Yields
        ------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            A generator yielding tuples of training indices and validation indices for each fold.
        """
        for fold in data[self.fold_col].unique():
            train_idx, val_idx = self.get_fold_idx(data, fold)
            yield train_idx.to_numpy(), val_idx.to_numpy()

    def get_fold(self, data: pd.DataFrame,
                 fold: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get the training and validation data for a specific fold.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing features and labels.
        fold : int
            The fold number to retrieve the training and validation data for.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple containing the training features, training labels,
            validation features, and validation labels for the specified fold.
        """
        X_train = data.loc[data[self.fold_col] != fold,]
        y_train = data.loc[data[self.fold_col] != fold]
        X_val = data.loc[data[self.fold_col] == fold]
        y_val = data.loc[data[self.fold_col] == fold]
        return (X_train, y_train), (X_val, y_val)

    def split_df(self, data: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate training and validation data for each fold.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing features and labels.

        Yields
        ------
        Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]
            A generator yielding tuples of training features, training labels,
            validation features, and validation labels for each fold.
        """
        for fold in data[self.fold_col].unique():
            yield self.get_fold(data, fold)
