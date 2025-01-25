from typing import List, Optional, Tuple
import ROOT
import pandas as pd


def rdf_column_names(rdf: ROOT.RDataFrame) -> List[str]:
    """
    Get the column names of a RDataFrame.

    Parameters
    ----------
    rdf : ROOT.RDataFrame
        The RDataFrame object.

    Returns
    -------
    List[str]
        The column names of the RDataFrame.
    """
    return [str(col) for col in rdf.GetColumnNames()]


def rdf_to_pandas(rdf: ROOT.RDataFrame,
                  columns: Optional[List[str]] = None,
                  nrows: int = -1) -> pd.DataFrame:
    """
    Convert a RDataFrame to a pandas DataFrame.

    Parameters
    ----------
    rdf : ROOT.RDataFrame
        The RDataFrame object.
    columns: Optional[List[str]]
        The columns to convert. By default, all columns are converted.
    nrows : int, optional
        Number of rows to convert. By default, all rows are converted.
        This param does not avoid loading all the data in memory, it just
        slices the data. By default -1.

    Returns
    -------
    pd.DataFrame
        The pandas DataFrame.
    """
    if columns is None:
        pdf_dict = rdf.AsNumpy()
    else:
        pdf_dict = rdf.AsNumpy(columns)
    keys = list(pdf_dict.keys())
    for key in keys:
        # this is necessary because the keys are of type
        # std::string (cppyy.gbl.std.string)
        # and pandas don't know how to deal with it
        if nrows < 0:
            pdf_dict[str(key)] = pdf_dict.pop(key)
        else:
            pdf_dict[str(key)] = pdf_dict.pop(key)[:nrows]
    return pd.DataFrame.from_dict(pdf_dict)


def open_vector(column: str, vec_len: int, rdf: ROOT.RDataFrame
                ) -> Tuple[List[str], ROOT.RDataFrame]:
    """
    Open a vector "vec" in multiple columns "vec_i".
    i is the idx of that value in "vec"

    Parameters
    ----------
    column : str
        The column name.
    vec_len : int
        The length of the vector.
    rdf : ROOT.RDataFrame
        The RDataFrame object.

    Returns
    -------
    List[str]
        List of the created columns.
    ROOT.RDataFrame
        The RDataFrame object with the new columns.
    """
    column_names = []
    for i in range(vec_len):
        new_name = f'{column}_{i}'
        rdf = rdf.Define(new_name, f'{column}[{i}]')
        column_names.append(new_name)
    return column_names, rdf

def parse_inf(value: str) -> str:
    """
    Parse a string that can be 'inf' to 1000000000 (1 billion).
    This is needed since ROOT does not support 'inf' as a value.

    Parameters
    ----------
    value : str
        The value to parse.

    Returns
    -------
    str
        The parsed value.
    """
    if value == 'inf':
        return '1000000000'
    elif value == '-inf':
        return '-1000000000'
    else:
        return value
