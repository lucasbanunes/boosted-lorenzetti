from pathlib import Path
import pandas as pd
from boosted_lorenzetti.cutbased.models import ElectronCutBasedModel, ElectronCutMap, L1Info
from boosted_lorenzetti.cutbased import cli


def test_init_cutbased_model(test_data_dir: Path):
    model = ElectronCutBasedModel(
        cut_map=ElectronCutMap(cuts='vloose'),
        l1=L1Info(
            eta_col='trig_L1eFex_el_eta',
            phi_col='trig_L1eFex_el_phi'
        )
    )
    test_df = pd.read_parquet(
        test_data_dir / 'test_cutbased' / 'test_parquet.parquet')
    result_df = model.predict(test_df)
    assert not result_df.empty, "Prediction result is empty."
    assert 'classification' in result_df.columns, \
        "'classification' column not found in the result."
    assert len(result_df) == len(test_df), \
        "Result length does not match input data length."


def test_predict(test_data_dir: Path):
    job = cli.predict(
        config_path=test_data_dir / 'test_cutbased' / 'predict.yaml'
    )
    dataset_path = job.dataset_path
    output_path = job.output.path
    assert output_path.exists(), "Output path does not exist."
    output_df = pd.read_parquet(output_path)
    dataset = pd.read_parquet(dataset_path)
    assert not output_df.empty, "Output DataFrame is empty."
    assert len(output_df) == len(dataset), \
        "Output DataFrame length does not match dataset length."
