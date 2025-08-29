import pytest
from pathlib import Path
import tempfile
from typing import Generator
import shutil


@pytest.fixture()
def repo_path() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture()
def test_data_dir(repo_path: Path) -> Path:
    return Path(__file__).parent / 'data'


@pytest.fixture()
def session_tmp_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture()
def n_folds() -> int:
    return 5


@pytest.fixture()
def test_dataset_path(test_data_dir: Path) -> Path:
    return test_data_dir / 'test_dataset.duckdb'


@pytest.fixture()
def test_npz_dataset_dir(test_data_dir: Path) -> Path:
    return test_data_dir / 'npz_dataset'


@pytest.fixture()
def test_zee_parquet_dataset_dir(test_data_dir: Path) -> Path:
    return test_data_dir / 'zee_parquet_dataset'


@pytest.fixture()
def test_duckdb_dataset(test_data_dir: Path) -> Path:
    return test_data_dir / 'test_dataset.duckdb'


@pytest.fixture()
def writable_test_duckdb_dataset(test_duckdb_dataset: Path,
                                 tmp_path: Path) -> Path:
    writable_path = tmp_path / 'writable_test_duckdb_dataset.duckdb'
    shutil.copy(test_duckdb_dataset, writable_path)
    return writable_path
