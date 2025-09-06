from pathlib import Path
import shutil
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
                  output_file=output_file,
                  ttree_name='CollectionTree',
                  batch_size=10)
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
                             '--output-file', str(output_file),
                             '--batch-size', '10'],
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


def test_aod_create_ringer_dataset(test_data_dir: Path,
                                   tmp_path: Path):
    test_file = test_data_dir / 'test.AOD.duckdb'
    second_test_file = tmp_path / 'test.AOD_copy.duckdb'
    shutil.copy(test_data_dir / 'test.AOD.duckdb', second_test_file)
    output_ringer_file = tmp_path / 'test_ringer_dataset.duckdb'

    # Test create_ringer_dataset function
    aod.create_ringer_dataset(
        input_dbs=f'{str(test_file)}, {str(second_test_file)}',
        output_file=output_ringer_file,
        labels="0, 1",
        description="Test ringer dataset"
    )

    assert output_ringer_file.exists(), "Ringer dataset file does not exist."

    # Testing if created ringer dataset is readable
    with duckdb.connect(str(output_ringer_file)) as con:
        # Check if data table exists and has data
        df = con.execute("SELECT * FROM data LIMIT 1").pl()
        assert len(df) > 0, "Data table is empty or unreadable."

        # Check if sources table exists
        df_source = con.execute("SELECT * FROM sources").pl()
        assert len(df_source) >= 0, "sources table is unreadable."

        # Check if metadata table exists
        df_metadata = con.execute("SELECT * FROM metadata").pl()
        assert len(df_metadata) > 0, "Metadata table is empty or unreadable."


def test_aod_create_ringer_dataset_cli(test_data_dir: Path,
                                       repo_path: Path,
                                       tmp_path: Path):
    test_file = test_data_dir / 'test.AOD.duckdb'
    second_test_file = tmp_path / 'test.AOD_copy.duckdb'
    shutil.copy(test_data_dir / 'test.AOD.duckdb', second_test_file)
    output_ringer_file = tmp_path / 'test_ringer_dataset.duckdb'

    # Test CLI command
    result = subprocess.run(['python',
                             f'{str(repo_path)}/cli.py',
                             'aod',
                             'create-ringer-dataset',
                             '--input-dbs', f'{str(test_file)}, {str(second_test_file)}',
                             '--output-file', str(output_ringer_file),
                             '--labels', '0, 1',
                             '--description', 'Test ringer dataset'],
                            capture_output=True, text=True)
    logging.info("STDOUT: %s", result.stdout )
    logging.error("STDERR: %s", result.stderr)

    assert output_ringer_file.exists(), "Ringer dataset file does not exist."

    # Testing if created ringer dataset is readable
    with duckdb.connect(str(output_ringer_file)) as con:
        # Check if data table exists and has data
        df = con.execute("SELECT * FROM data LIMIT 1").pl()
        assert len(df) > 0, "Data table is empty or unreadable."

        # Check if sources table exists
        df_source = con.execute("SELECT * FROM sources").pl()
        assert len(df_source) >= 0, "sources table is unreadable."

        # Check if metadata table exists
        df_metadata = con.execute("SELECT * FROM metadata").pl()
        assert len(df_metadata) > 0, "Metadata table is empty or unreadable."
