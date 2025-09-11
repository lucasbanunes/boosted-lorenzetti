from pathlib import Path
import duckdb
import subprocess
import logging

from boosted_lorenzetti.dataset import npz


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
    logging.info("Running NPZ to DuckDB conversion via CLI")
    logging.info(f"Dataset dir: {test_npz_dataset_dir}\n"
                 f"Output file: {output_file}")
    result = subprocess.run(['python',
                             f'{str(repo_path)}/cli.py',
                             'npz',
                             'to-duckdb',
                             '--dataset-dir', str(test_npz_dataset_dir),
                             '--output-file', str(output_file)],
                            capture_output=True, text=True)
    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)
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
