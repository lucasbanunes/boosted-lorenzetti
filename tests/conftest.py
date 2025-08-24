import pytest
from pathlib import Path
import tempfile
from typing import Generator


@pytest.fixture(scope='session')
def repo_path() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope='session')
def test_data_dir(repo_path: Path) -> Path:
    return Path(__file__).parent / 'data'


@pytest.fixture(scope='session')
def session_tmp_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope='session')
def n_folds() -> int:
    return 5


@pytest.fixture(scope='session')
def test_dataset_path(test_data_dir: Path) -> Path:
    return test_data_dir / 'test_dataset.duckdb'


@pytest.fixture(scope='session')
def test_npz_dataset_dir(test_data_dir: Path) -> Path:
    return test_data_dir / 'npz_dataset'


@pytest.fixture(scope='session')
def test_zee_parquet_dataset_dir(test_data_dir: Path) -> Path:
    return test_data_dir / 'zee_parquet_dataset'
