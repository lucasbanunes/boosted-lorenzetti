from pathlib import Path
import pandas as pd

from boosted_lorenzetti.dataset.convert import convert
from boosted_lorenzetti.dataset.ingest import ingest


def test_convert_ntuple2parquet(test_data_dir: Path,
                                tmp_path: Path):
    test_file = test_data_dir / 'test.NTUPLE.root'
    convert([test_file],
            input_format='ntuple',
            output_format='parquet',
            output_dir=tmp_path,
            n_jobs=1)
    output_file = tmp_path / 'test.NTUPLE.parquet'
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    assert pd.read_parquet(
        output_file).shape[0] > 0, "Parquet file is empty or unreadable."


def test_ingest(test_data_dir: Path,
                tmp_path: Path):
    electron_dataset = test_data_dir / 'zee_dataset'
    jet_dataset = test_data_dir / 'jf17_dataset'
    output_dir = tmp_path / 'ingest_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, output_file = ingest(electron_dataset=str(electron_dataset),
                               jet_dataset=str(jet_dataset),
                               name='test_ingest',
                               output_dir=output_dir,
                               lzt_version='1.0.0',
                               n_folds=2,
                               seed=42)

    assert output_file.exists(), "Ingested dataset does not exist."
    assert pd.read_parquet(
        output_file).shape[0] > 0, "Ingested dataset is empty or unreadable."
