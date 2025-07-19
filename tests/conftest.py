import pytest
from pathlib import Path
import tempfile


@pytest.fixture(scope='session')
def test_data_dir():
    return Path(__file__).parent / 'data'


@pytest.fixture
def test_temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
