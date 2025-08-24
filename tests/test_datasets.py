from pathlib import Path
import pandas as pd
import duckdb
import subprocess

from boosted_lorenzetti.dataset import ntuple, npz
from boosted_lorenzetti.dataset import duckdb as boosted_duckdb


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


def test_create_dataset(test_data_dir: Path,
                        tmp_path: Path):
    electron_dataset = test_data_dir / 'zee_dataset'
    jet_dataset = test_data_dir / 'jf17_dataset'
    output_path = tmp_path / 'test_create_dataset.duckdb'
    table_name = 'ntuple'
    output_file = ntuple.create_dataset(
        ntuple_paths=[
            electron_dataset,
            jet_dataset
        ],
        labels=[1, 0],
        output_path=output_path,
        lzt_version='vTest',
        n_folds=2,
        seed=42,
        table_name=table_name,
        experiment_name='text_create_dataset',
    )

    assert output_file.exists(), "Created dataset does not exist."
    # Testing if created dataset is readable
    with duckdb.connect(str(output_file)) as con:
        df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    assert len(df) > 0, "Created dataset is empty or unreadable."


def test_npz_to_duckdb(test_npz_dataset_dir: Path,
                       tmp_path: Path):
    output_file = tmp_path / 'test_npz_to_duckdb.duckdb'
    npz.to_duckdb(dataset_dir=test_npz_dataset_dir,
                  output_file=output_file,
                  overwrite=True)
    assert output_file.exists(), "Converted DuckDB file does not exist."
    # Testing if converted format is readable
    with duckdb.connect(str(output_file)) as con:
        data_df = con.execute("SELECT * FROM data;").pl()
        assert len(data_df) > 0, "DuckDB file is empty or unreadable."
        assert len(data_df['id'].unique()) == len(
            data_df), "data_df ids are not unique"
        references_df = con.execute(
            "SELECT * FROM model_references").pl()
        assert len(references_df) > 0, "DuckDB file is empty or unreadable."
        assert len(references_df['id'].unique()) == len(
            references_df), "references_df ids are not unique"


def test_npz_to_duckdb_cli(test_npz_dataset_dir: Path,
                           repo_path: Path,
                           tmp_path: Path):
    output_file = tmp_path / 'test_npz_to_duckdb_cli.duckdb'
    subprocess.run(['python',
                    f'{str(repo_path)}/cli.py',
                    'npz',
                    'to-duckdb',
                    '--dataset-dir', str(test_npz_dataset_dir),
                    '--output-file', str(output_file)])
    assert output_file.exists(), "Converted DuckDB file does not exist."
    # Testing if converted format is readable
    with duckdb.connect(str(output_file)) as con:
        data_df = con.execute("SELECT * FROM data;").pl()
        assert len(data_df) > 0, "DuckDB file is empty or unreadable."
        assert len(data_df['id'].unique()) == len(
            data_df), "data_df ids are not unique"
        references_df = con.execute(
            "SELECT * FROM model_references").pl()
        assert len(references_df) > 0, "DuckDB file is empty or unreadable."
        assert len(references_df['id'].unique()) == len(
            references_df), "references_df ids are not unique"


def test_duckdb_add_table_from_parquet(test_zee_parquet_dataset_dir: Path,
                                       tmp_path: Path):
    output_file = tmp_path / 'test_duckdb_add_table_from_parquet.duckdb'
    boosted_duckdb.add_table_from_parquet(
        db_path=str(output_file),
        table_name='physics',
        files=[str(test_zee_parquet_dataset_dir) + '/*.parquet']
    )
    assert output_file.exists(), "Converted DuckDB file does not exist."
    # Testing if converted format is readable
    with duckdb.connect(str(output_file)) as con:
        df = con.execute("SELECT * FROM physics").pl()
    assert len(df) > 0, "DuckDB file is empty or unreadable."
