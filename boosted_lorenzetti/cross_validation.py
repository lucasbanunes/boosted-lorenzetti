from typing import List, Generator, Tuple
import pandas as pd


class ColumnKFold:
    def __init__(self,
                 fold_col: str,
                 feature_cols: List[str],
                 label_cols: List[str] = ['label'],):
        self.fold_col = fold_col
        self.feature_cols = feature_cols
        self.label_cols = label_cols

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
        X_train = data.loc[data[self.fold_col] != fold, self.feature_cols]
        y_train = data.loc[data[self.fold_col] != fold, self.label_cols]
        X_val = data.loc[data[self.fold_col] == fold, self.feature_cols]
        y_val = data.loc[data[self.fold_col] == fold, self.label_cols]
        return (X_train, y_train), (X_val, y_val)

    def split(self, data: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame], None, None]:
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
