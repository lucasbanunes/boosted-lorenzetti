from pathlib import Path
import pandas as pd

from boosted_lorenzetti.dataset.convert import convert


def test_convert_ntuple2parquet(test_data_dir: Path,
                                test_temp_dir: Path):
    test_file = test_data_dir / 'test.NTUPLE.root'
    convert([test_file],
            input_format='ntuple',
            output_format='parquet',
            output_dir=test_temp_dir,
            n_jobs=1)
    output_file = test_temp_dir / 'test.NTUPLE.parquet'
    assert output_file.exists(), "Converted file does not exist."

    # Testing if converted format is readable
    assert pd.read_parquet(
        output_file).shape[0] > 0, "Parquet file is empty or unreadable."
