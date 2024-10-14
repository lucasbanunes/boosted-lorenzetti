import os
import argparse
from typing import List
from lzt_utils.utils import open_directories
from lxt_utils.root import parse_inf, rdf_column_names
from itertools import product

def parse_args() -> dict:
    parser = ArgumentParser(
        prog=os.path.basename(__file__),
        description="Converts a TTree in a group of root files "
        "to a desired file format."
    )
    parser.add_argument(
        '--filenames',
        required=True,
        nargs='+',
        help='Path to the files to be converted.'
        'If path is a dir, the script looks for .root files'
    )
    parser.add_argument(
        '--treepath',
        required=True,
        help='The path to the tree object inside the root file'
    )
    parser.add_argument(
        '--output_dir',
        dest='output_dir',
        required=True,
        help='Directory to save the resulting files'
    )
    parser.add_argument(
        '--et-col',
        required=True,
        dest='et_col',
        type=str,
        help='Name of the et column to split on'
    )
    parser.add_argument(
        '--et-bins',
        required=True,
        dest='et_bins',
        nargs='+',
        type=float,
        help='Bins to split the data in et'
    )
    parser.add_argument(
        '--eta-col',
        required=True,
        dest='eta_col',
        type=str,
        help='Name of the eta column to split on'
    )
    parser.add_argument(
        '--eta-bins',
        required=True,
        dest='eta_bins',
        nargs='+',
        type=float,
        help='Bins to split the data in eta. It uses |eta| to ensure symetry.'
    )
    parser.add_argument(
        '--include-cols',
        nargs='+',
        default=None,
        dest='include_cols',
        help='List of columns to be included in the export. If not passed, includes all.'
    )
    parser.add_argument(
        '--exclude-cols',
        nargs='+',
        default=None,
        dest='exclude_cols',
        help='List of columns to be excluded in the export. If not passed, excludes None.'
    )
    args = parser.parse_args()
    return args.__dict__


def bin_data_in_et_eta(filenames: List[str],
                       treepath: str,
                       output_dir: str,
                       et_col: str,
                       et_bins: List[float],
                       eta_col: str,
                       eta_bins: List[float],
                       include_cols: List[str],
                       exclude_cols: List[str]) -> None:
    # Load data
    filenames = open_directories(filenames, 'root')
    rdf = ROOT.RDataFrame(treepath, filenames)

    parsed_et_bins = [parse_inf(et) for et in et_bins]
    parsed_eta_bins = [parse_inf(eta) for eta in eta_bins]
    iterator = product(range(len(et_bins)-1), range(len(eta_bins)-1))

    export_columns = rdf_column_names(rdf)
    if include_columns is not None:
        export_columns = filter(lambda x: x in include_columns, export_columns)
    if exclude_columns is not None:
        export_columns = filter(lambda x: x not in exclude_columns, export_columns)
    export_columns = list(export_columns)

    os.makedirs(output_dir, exist_ok=True)

    for et_idx, eta_idx in iterator:
        parsed_et_min = parsed_et_bins[et_idx]
        parsed_et_max = parsed_et_bins[et_idx+1]
        parsed_eta_min = parsed_eta_bins[eta_idx]
        parsed_eta_max = parsed_eta_bins[eta_idx+1]
        data = rdf.Filter(f'{et_col} >= {parsed_et_min} && {et_col} < {parsed_et_max}'
                          f' && {eta_col} >= {parsed_eta_min} && {eta_col} < {parsed_eta_max}')
        data = data.AsNumpy(columns=export_columns)
        pdf = pd.DataFrame.from_dict(data)
        pdf.to_parquet(f'{filename}_et_{et_min}_{et_max}_eta_{eta_min}_{eta_max}.parquet')
        save_data(data, filename, output_ext)