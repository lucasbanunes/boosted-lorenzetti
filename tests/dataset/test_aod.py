from pathlib import Path
import pandas as pd
import subprocess
import logging
import duckdb

from boosted_lorenzetti.dataset import aod


def test_aod_to_parquet(test_data_dir: Path,
                        tmp_path: Path):
    test_file = test_data_dir / 'test.AOD.root'
    output_file = tmp_path / 'test.NTUPLE.parquet'
    aod.to_parquet(str(test_file),
                   output_file=str(output_file),
                   ttree_name='CollectionTree')
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    assert pd.read_parquet(
        output_file, dtype_backend='pyarrow').shape[0] > 0, "Parquet file is empty or unreadable."


def test_aod_to_parquet_cli(test_data_dir: Path,
                            repo_path: Path,
                            tmp_path: Path):
    test_file = test_data_dir / 'test.AOD.root'
    output_file = tmp_path / 'test.NTUPLE.parquet'
    result = subprocess.run(['python',
                             f'{str(repo_path)}/cli.py',
                             'aod',
                             'to-parquet',
                             '--input-file', str(test_file),
                             '--output-file', str(output_file)],
                            capture_output=True, text=True)
    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    assert pd.read_parquet(
        output_file, dtype_backend='pyarrow').shape[0] > 0, "Parquet file is empty or unreadable."


def test_aod_to_duckdb(test_data_dir: Path,
                       tmp_path: Path):
    test_file = test_data_dir / 'test.AOD.root'
    output_file = tmp_path / 'test.AOD.duckdb'
    aod.to_duckdb(str(test_file),
                  output_file=str(output_file),
                  ttree_name='CollectionTree')
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    with duckdb.connect(str(output_file)) as con:
        df = con.execute("SELECT * FROM clusters").pl()
        assert len(df) > 0, "Clusters table is empty or unreadable."
        df = con.execute("SELECT * FROM events").pl()
        assert len(df) > 0, "Events table is empty or unreadable."


def test_aod_to_duckdb_cli(test_data_dir: Path,
                           repo_path: Path,
                           tmp_path: Path):
    test_file = test_data_dir / 'test.AOD.root'
    output_file = tmp_path / 'test.AOD.duckdb'
    result = subprocess.run(['python',
                             f'{str(repo_path)}/cli.py',
                             'aod',
                             'to-duckdb',
                             '--input-file', str(test_file),
                             '--output-file', str(output_file)],
                            capture_output=True, text=True)
    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    with duckdb.connect(str(output_file)) as con:
        df = con.execute("SELECT * FROM clusters").pl()
        assert len(df) > 0, "Clusters table is empty or unreadable."
        df = con.execute("SELECT * FROM events").pl()
        assert len(df) > 0, "Events table is empty or unreadable."
