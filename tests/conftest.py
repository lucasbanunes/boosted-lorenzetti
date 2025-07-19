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


@pytest.fixture
def test_temp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
