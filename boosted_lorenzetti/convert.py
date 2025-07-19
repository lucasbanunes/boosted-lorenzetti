from typer import Typer
from pathlib import Path
from typing import List, Literal
import pyarrow as pa

app = Typer()


def ntuple_to_parquet(input_path: Path | str,
                      output_path: Path | str):
    


type SupportedInputs = Literal['ntuple']

type SupportedOutputs = Literal['parquet']


@app_command()
def convert(input_files: Path | List[Path],
            output_dir: Path | str,
            input_format: SupportedInputs,
            output_format: SupportedOutputs):
    if input_format != 'ntuple':
        raise ValueError(f"Unsupported input format: {input_format}. Supported formats: 'ntuple'.")
    if output_format != 'parquet':
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats: 'parquet'.")
    
    raise NotImplementedError("Conversion functionality is not yet implemented.")