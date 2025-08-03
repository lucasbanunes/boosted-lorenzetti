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
    # test_dataset_path = session_tmp_path / 'test_dataset_path'
    # df_dir = test_dataset_path / 'data'
    # df_dir.mkdir(parents=True, exist_ok=True)
    # df_path = df_dir / 'data_test.parquet'
    # # Create a dummy dataset
    # size = n_folds * 100  # 100 samples per fold
    # df = {
    #     f'ring_{i}': np.random.rand(size) for i in range(N_RINGS)
    # }
    # df['label'] = np.random.randint(0, 2, size=size)
    # df['fold'] = np.random.randint(0, n_folds, size=size)
    # df['id'] = np.arange(1, size + 1)
    # df = pd.DataFrame(df)
    # df.to_parquet(df_path)

    # return test_dataset_path
    return test_data_dir / 'test_dataset.duckdb'
