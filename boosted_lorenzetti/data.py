import pandas as pd
from torch.utils.data import TensorDataset, Dataset
from typing import List
import torch
import hashlib


def tensor_dataset_from_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_cols: List[str] = ['label'],
    feature_type: str = 'float32',
    label_type: str = 'int64',
    idx: List[int] | None = None
) -> Dataset:
    """
    Converts a pandas DataFrame into a PyTorch Dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataframe
    feature_cols : List[str]
        Columns to be used as features.
    label_cols : List[str], optional
        Columns to be used as labels, by default ['label']
    feature_type : str, optional
        Data type for the features, by default 'float32'
    label_type : str, optional
        Data type for the labels, by default 'int64'
    idx : List[int] | None, optional
        Indices to select a subset of the DataFrame, by default None

    Returns
    -------
    Dataset
        A PyTorch Dataset containing the features and labels.
    """
    if idx is not None:
        features = df.loc[idx, feature_cols].values.astype(feature_type)
        labels = df.loc[idx, label_cols].values.astype(label_type)
    else:
        features = df[feature_cols].values.astype(feature_type)
        labels = df[label_cols].values.astype(label_type)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(labels),
    )
    return dataset


def get_dataframe_hash(df: pd.DataFrame) -> str:
    """
    Computes a hash for the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to hash.

    Returns
    -------
    str
        A SHA-256 hash of the DataFrame.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=False)
    return hashlib.sha256(row_hashes.values).hexdigest()
