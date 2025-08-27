import numpy as np
import cyclopts
from pydantic import Field
from typing import Annotated, List
from pathlib import Path


def seed_factory() -> int:
    return np.random.randint(np.iinfo(np.int32).max - 10)


SEED_DESCRIPTION = "Random seed for reproducibility. If not provided, a random seed will be generated."

SeedType = Annotated[
    int,
    Field(
        default_factory=seed_factory,
        description=SEED_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--seed",
        help=SEED_DESCRIPTION,
    )
]

MLFLOWRUNID_DESCRIPTION = (
    "MLFlow run ID for tracking the training job. If not provided, a new run will be created."
)


MLFlowRunId = Annotated[
    str,
    Field(
        description=MLFLOWRUNID_DESCRIPTION
    ),
]


CHECKPOINTS_DIR_DESCRIPTION = "Directory to save model checkpoints."

CheckpointsDirType = Annotated[
    Path,
    Field(
        description=CHECKPOINTS_DIR_DESCRIPTION
    ),
    cyclopts.Parameter(
        help=CHECKPOINTS_DIR_DESCRIPTION,
    )
]

BATCH_SIZE_DESCRIPTION = (
    "Batch size for training the model."
)

BatchSizeType = Annotated[
    int,
    Field(
        description=BATCH_SIZE_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--batch-size",
        help=BATCH_SIZE_DESCRIPTION,
    )
]

DIMS_DESCRIPTION = 'Dimensions of the model layers.'

DimsType = Annotated[
    List[int],
    Field(
        description=DIMS_DESCRIPTION
    ),
    cyclopts.Parameter(
        help=DIMS_DESCRIPTION,
    )
]

ACTIVATION_DESCRIPTION = (
    "Activation function to use in the model."
)

ActivationType = Annotated[
    str,
    Field(
        description=ACTIVATION_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--activation",
        help=ACTIVATION_DESCRIPTION,
    )
]

DF_NAME_DESCRIPTION = (
    "Name of the DataFrame to use for training."
)

DfNameType = Annotated[
    str,
    Field(
        description=DF_NAME_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--df-name",
        help=DF_NAME_DESCRIPTION,
    )
]

FOLD_COLUMN_DESCRIPTION = (
    "Column name in the DataFrame that indicates the fold number for cross-validation."
)

FoldColType = Annotated[
    str,
    Field(
        default='fold',
        description=FOLD_COLUMN_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--fold-col",
        help=FOLD_COLUMN_DESCRIPTION,
    )
]

FEATURE_COLS_DESCRIPTION = (
    "List of feature column names in the DataFrame."
)

FeatureColsType = Annotated[
    List[str],
    Field(
        description=FEATURE_COLS_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--feature-cols",
        help=FEATURE_COLS_DESCRIPTION,
    )
]

LABEL_COLS_DESCRIPTION = (
    "List of label column names in the DataFrame."
)

LabelColsType = Annotated[
    List[str],
    Field(
        description=LABEL_COLS_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--label",
        help=LABEL_COLS_DESCRIPTION,
    )
]

INIT_DESCRIPTION = (
    "Initialization identifier for the model parameters."
)

InitType = Annotated[
    int,
    Field(
        description=INIT_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--init",
        help=INIT_DESCRIPTION,
    )
]

FOLD_DESCRTIPTION = (
    "Fold number for cross-validation."
)

FoldType = Annotated[
    int,
    Field(
        description=FOLD_DESCRTIPTION
    ),
    cyclopts.Parameter(
        "--fold",
        help=FOLD_DESCRTIPTION,
    )
]

ACCELERATOR_DESCRIPTION = (
    "Type of accelerator to use for training."
)

AcceleratorType = Annotated[
    str,
    Field(
        description=ACCELERATOR_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--accelerator",
        help=ACCELERATOR_DESCRIPTION,
    )
]

PATEINCE_DESCRIPTION = (
    "Number of epochs with no improvement after which training will be stopped."
)

PatienceType = Annotated[
    int,
    Field(
        description=PATEINCE_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--patience",
        help=PATEINCE_DESCRIPTION,
    )
]

INITS_DESCRIPTION = (
    "Number of different initializations to perform for the model."
)

InitsType = Annotated[
    int,
    Field(
        description=INITS_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--inits",
        help=INITS_DESCRIPTION,
    )
]

TRACKING_URI_DESCRIPTION = (
    "MLFlow tracking URI for logging the training job.."
)

TrackingUriType = Annotated[
    str,
    Field(
        description=TRACKING_URI_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--tracking-uri",
        help=TRACKING_URI_DESCRIPTION,
    )
]

EXPERIMENT_NAME_DESCRIPTION = (
    "Name of the MLFlow experiment to log the training job."
)

ExperimentNameType = Annotated[
    str | None,
    Field(
        description=EXPERIMENT_NAME_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--experiment-name",
        help=EXPERIMENT_NAME_DESCRIPTION,
    )
]

JOB_NAME_DESCRIPTION = (
    "Name of the training job."
)


MAX_EPOCHS_DESCRIPTION = (
    "Maximum number of epochs for training."
)

MaxEpochsType = Annotated[
    int,
    Field(
        description=MAX_EPOCHS_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--max-epochs",
        help=MAX_EPOCHS_DESCRIPTION,
    )
]

BEST_METRIC_DESCRIPTION = (
    "Best metric to use for KFold."
)

BestMetricType = Annotated[
    str,
    Field(
        description=BEST_METRIC_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--best-metric",
        help=BEST_METRIC_DESCRIPTION,
    )
]

BEST_METRIC_MODE_DESCRIPTION = (
    "Mode of the best metric to use for KFold."
)

BestMetricModeType = Annotated[
    str,
    Field(
        description=BEST_METRIC_MODE_DESCRIPTION
    ),
    cyclopts.Parameter(
        "--best-metric-mode",
        help=BEST_METRIC_MODE_DESCRIPTION,
    )
]
