from pathlib import Path
from typing import Annotated, Generator
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import yaml

from .models import ElectronCutBasedModel
from ..utils import open_directories
from ..log import get_logger
from ..types import DevType


class ElectronCutBasedJobOutputparams(BaseModel):

    path: Annotated[
        Path,
        Field(
            description="Output path. If dataset is a directory, it should be a directory too."
        )
    ]
    concat: Annotated[
        bool,
        Field(
            description="If True and output path is a file, all results will be concatenated into a single file."
        )
    ] = False


class ElectronCutBasedJob(BaseModel):

    dataset_path: Annotated[
        Path,
        Field(
            description="Parquet dataset to be used. If a directory is given, all parquet files in it will be processed."
        )
    ]
    dev: DevType = False
    models: Annotated[
        dict[str, ElectronCutBasedModel],
        Field(
            description="Cut based models to be used"
        )
    ]
    n_jobs: Annotated[
        int,
        Field(
            description="Number of parallel jobs to be used. Works if dataset is a directory. If negative, uses all available cores."
        )
    ] = 1
    output: Annotated[
        ElectronCutBasedJobOutputparams,
        Field(
            description="Output arguments for the job."
        )
    ]
    return_reason: Annotated[
        bool,
        Field(
            description="If True, the output will contain the reason for the classification."
        )
    ] = False
    verbose: Annotated[
        int,
        Field(
            description="Verbosity level for logging."
        )
    ] = 51

    def model_post_init(self, context):
        res = super().model_post_init(context)

        if self.dataset_path.is_dir():
            self.output.path.mkdir(parents=True, exist_ok=True)

        return res

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ElectronCutBasedJob":
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def predict_args(self) -> Generator[tuple[Path, Path], None, None]:
        if not self.dataset_path.is_dir():
            yield self.dataset_path, self.output.path
            return

        for filepath in open_directories(
            self.dataset_path,
            file_ext='parquet',
            dev=self.dev
        ):
            yield filepath, self.output.path / filepath.name

    def job(self, filepath: Path, output: Path, job_id: int):
        logger = get_logger()
        logger.info(f'{job_id} - Processing file: {filepath}')
        df = pd.read_parquet(filepath)
        results = []
        for model_name, model in self.models.items():
            logger.info(f'{job_id} - Predicting with model: {model_name}')
            rename_dict = {
                model.classification_col_name: f'{model_name}_{model.classification_col_name}',
                model.reason_col_name: f'{model_name}_{model.reason_col_name}'
            }
            result_df = model.predict(df, self.return_reason).rename(columns=rename_dict)
            results.append(result_df)
        if self.output.concat:
            results.insert(0, df)
        result_df = pd.concat(results, axis=1)
        logger.info(f'{job_id} - Saving results to: {output}')
        result_df.to_parquet(output)

    def run(self):
        pool = joblib.Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        pool(
            joblib.delayed(self.job)(filepath, output_path, job_id)
            for job_id, (filepath, output_path) in enumerate(self.predict_args())
        )
