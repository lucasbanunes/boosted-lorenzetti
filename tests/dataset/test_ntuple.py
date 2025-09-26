from pathlib import Path
import pandas as pd
import duckdb

from boosted_lorenzetti.dataset import ntuple


def test_convert_ntuple_to_parquet(test_data_dir: Path,
                                   tmp_path: Path):
    test_file = test_data_dir / 'test.NTUPLE.root'
    output_file = tmp_path / 'test.NTUPLE.parquet'
    ntuple.to_parquet(str(test_file),
                      output_file=str(output_file),
                      ttree_name='physics')
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    assert pd.read_parquet(
        output_file, dtype_backend='pyarrow').shape[0] > 0, "Parquet file is empty or unreadable."


def test_convert_ntuple_to_duckdb(test_data_dir: Path,
                                  tmp_path: Path):
    test_file = test_data_dir / 'test.NTUPLE.root'
    output_file = tmp_path / 'test.NTUPLE.duckdb'
    ntuple.to_duckdb(input_file=str(test_file),
                     output_file=str(output_file),
                     ttree_name='physics',
                     table_name='physics',
                     open_vectors=False)
    assert output_file.exists(), "Converted DuckDB file does not exist."
    # Testing if converted format is readable
    with duckdb.connect(str(output_file)) as con:
        df = con.execute("SELECT * FROM physics").pl()
    assert len(df) > 0, "DuckDB file is empty or unreadable."
