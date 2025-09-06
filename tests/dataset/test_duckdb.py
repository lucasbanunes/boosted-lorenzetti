from pathlib import Path
import pandas as pd
import duckdb
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import logging
import subprocess


from boosted_lorenzetti.dataset import duckdb as boosted_duckdb


CLASS_WEIGHTS_QUERY = """
SELECT
    label,
    COUNT(label) as n_labels,
    (SELECT (SELECT COUNT(*) FROM test_data WHERE label >=0)/COUNT(DISTINCT(label)) FROM test_data WHERE label >= 0)/n_labels as weights
FROM test_data
GROUP BY label HAVING label >= 0;"""


def test_balanced_class_weights(tmp_path: Path):
    df = pd.DataFrame()
    df['label'] = np.random.choice([0, 1], size=1000)

    with duckdb.connect(database=':memory:') as conn:
        conn.register('test_data', df)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(df['label']),
            y=df['label'].values
        )
        duckdb_weights = boosted_duckdb.get_balanced_class_weights(conn, 'test_data', 'label')

        assert np.allclose(class_weights, duckdb_weights,
                           rtol=1e-6), "Class weights do not match between sklearn and DuckDB."


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


def test_add_kfold(writable_test_duckdb_dataset: Path, n_folds: int):
    src_table = 'data'
    fold_col = 'test_fold'
    boosted_duckdb.add_kfold(
        db_path=str(writable_test_duckdb_dataset),
        label_col='label',
        src_table=src_table,
        n_folds=n_folds,
        fold_col=fold_col
    )
    assert writable_test_duckdb_dataset.exists(), "KFold DuckDB file does not exist."
    # Testing if converted format is readable
    with duckdb.connect(str(writable_test_duckdb_dataset)) as con:
        df = con.execute(f"SELECT {fold_col} FROM {src_table}").pl()
        assert len(df) > 0, "KFold DuckDB file is empty or unreadable."


def test_add_kfold_cli(writable_test_duckdb_dataset: Path, n_folds: int,
                       repo_path: Path):
    src_table = 'data'
    fold_col = 'test_fold'
    command = [
        'python',
        f'{str(repo_path)}/cli.py',
        'duckdb', 'add-kfold',
        '--db-path', str(writable_test_duckdb_dataset),
        '--label-col', 'label',
        '--src-table', src_table,
        '--n-folds', str(n_folds),
        '--fold-col', fold_col
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    logging.info("STDOUT: %s", result.stdout)
    logging.error("STDERR: %s", result.stderr)
    assert result.returncode == 0, "KFold CLI command failed."
    assert writable_test_duckdb_dataset.exists(), "KFold DuckDB file does not exist."
    # Testing if converted format is readable
    with duckdb.connect(str(writable_test_duckdb_dataset)) as con:
        df = con.execute(f"SELECT {fold_col} FROM {src_table}").pl()
        assert len(df) > 0, "KFold DuckDB file is empty or unreadable."
