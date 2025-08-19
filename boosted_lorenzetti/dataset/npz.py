from pathlib import Path
from typing import Annotated
import typer
import duckdb
import numpy as np
import polars as pl
from collections import defaultdict
import gzip
import pickle
from io import BytesIO
import logging

from .duckdb import check_table_exists


app = typer.Typer(
    name='npz',
    help='Operates with legacy npz datasets'
)

FEATURE_TYPES = ['int', 'float', 'object', 'bool']

MODEL_REFERENCES_SCHEMA = pl.Schema({
    'et_bin_lower': pl.Float32(),
    'et_bin_upper': pl.Float32(),
    'eta_bin_lower': pl.Float32(),
    'eta_bin_upper': pl.Float32(),
    'pid': pl.String(),
    'label': pl.UInt8(),
    'total': pl.Int64(),
    'passed': pl.Int64(),
    'reference': pl.String(),
})


@app.command(help='Converts legacy npz dataset to duckdb databases to make easier to manipulate the data.')
def to_duckdb(dataset_dir: Annotated[Path, typer.Option(..., help="Path to the input dataset directory.")],
              output_file: Annotated[Path, typer.Option(..., help="Path to the output DuckDB file.")],
              overwrite: bool = False):
    """
    Converts legacy npz dataset to `.duckdb` databases to make easier to manipulate the data.
    The dataset is expected to have the following structure
    - /dataset_name
        - file1.npz
        -  file2.npz
        - /references
            - file1.pic.gz
            - file2.pic.gz

    The data in the npz files is mapped to the `data` table and the `refrences` files are mapped to the `model_refrences` table.

    The `.npz` files must have the following structure:
    - `etBins`: Array of floats with the et binning limits
    - `etaBins`: Array of floats with the eta binning limits
    - `etBinIdx`: Index of the bin used for this data
    - `etaBinIdx`: Index of the bin used for this data
    - `ordered_features`: Ignored
    - `data_float`: 2D array of floats with the data
    - `data_bool`: 2D array of bools with the data
    - `data_int`: 2D array of ints with the data
    - `data_object`: 2D array of objects with the data
    - `features_float`: Array with the column names in `data_float`
    - `features_bool`: Array with the column names in `data_bool`
    - `features_int`: Array with the column names in `data_int`
    - `features_object`: Array with the column names in `data_object`
    - `target`: Array with the target
    - `protocol`: Ignored
    - `allow_pickle`: Ignored

    The `.pic.gz` files must have the following structure:
    - `class`: Ignored
    - `__module`: Ignored
    - `etBinIdx`: Array with 1 element representing the et bin used for this data
    - `etBins`: Array of floats with the et binning limits
    - `etaBinIdx`: Array with 1 element representing the eta bin used for this data
    - `etaBins`: Array of floats with the eta binning limits
    - `__version`: Ignored
    - `bkgRef`: Dict
    - `sgnRef`: Dict

    Parameters
    ----------
    dataset_dir : Path
        Directory containing the NPZ files.
    output_file : Path
        Path to the output DuckDB file.
    overwrite : bool, optional
        Whether to overwrite the output file if it exists, by default False
    """

    if overwrite:
        output_file.unlink(missing_ok=True)

    with duckdb.connect(str(output_file)) as con:
        sample_count = 0
        for i, filepath in enumerate(dataset_dir.glob('*.npz')):
            logging.info(f'Processing file {i}: {filepath}')
            npz_file = np.load(filepath)
            feature_dfs = []
            for feature_type in FEATURE_TYPES:
                array_key = f'data_{feature_type}'
                schema_key = f'features_{feature_type}'
                feature_dfs.append(
                    pl.from_numpy(
                        npz_file[array_key],
                        schema=npz_file[schema_key].tolist(),
                        orient='row'
                    )
                )
            feature_dfs.append(
                pl.from_numpy(
                    npz_file['target'],
                    schema=['target'],
                    orient='row'
                )
            )
            aux_df: pl.DataFrame = pl.concat(feature_dfs, how='horizontal')
            del feature_dfs
            if 'id' not in aux_df.columns:
                aux_df = aux_df.with_columns(
                    pl.Series('id', np.arange(sample_count, sample_count+len(aux_df)))
                )
            if i == 0 and not check_table_exists(con, 'data'):
                con.execute(
                    "CREATE TABLE IF NOT EXISTS data AS SELECT * FROM aux_df")
            else:
                con.execute("INSERT INTO data SELECT * FROM aux_df")
            sample_count += len(aux_df)
            del aux_df

    references_dir = dataset_dir / 'references'

    table_data = defaultdict(list)
    for i, filepath in enumerate(references_dir.glob('*.pic.gz')):
        # if i > 0:
        #     break
        print(f'Processing reference file {i + 1}: {filepath}')
        with gzip.open(filepath, 'rb') as f:
            data = pickle.load(BytesIO(f.read()))
        et_bin_idx = data['etBinIdx']
        eta_bin_idx = data['etaBinIdx']
        for label, data_type in enumerate(['bkgRef', 'sgnRef']):
            for pid, pid_data in data[data_type].items():
                table_data['et_bin_lower'].append(
                    float(data['etBins'][et_bin_idx]))
                table_data['et_bin_upper'].append(
                    float(data['etBins'][et_bin_idx + 1]))
                table_data['eta_bin_lower'].append(
                    float(data['etaBins'][eta_bin_idx]))
                table_data['eta_bin_upper'].append(
                    float(data['etaBins'][eta_bin_idx + 1]))
                table_data['pid'].append(str(pid))
                table_data['label'].append(int(label))
                table_data['total'].append(int(pid_data['total']))
                table_data['passed'].append(int(pid_data['passed']))
                table_data['reference'].append(str(pid_data['reference']))
    reference_df = pl.from_dict(
        dict(table_data),
        schema=MODEL_REFERENCES_SCHEMA
    )
    reference_df = reference_df.with_columns(
        pl.Series('id', np.arange(len(reference_df)))
    )

    with duckdb.connect(str(output_file)) as con:
        if not check_table_exists(con, 'model_references'):
            con.execute(
                "CREATE TABLE IF NOT EXISTS model_references AS SELECT * FROM reference_df")
        else:
            con.execute(
                "INSERT INTO model_references SELECT * FROM reference_df")
