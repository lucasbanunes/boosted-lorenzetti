from typer import Typer
from pathlib import Path
from typing import Generator, List, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm

from . import ntuple
from ..utils import list_by_pattern


def file_generator(input_files: List[Path],
                   pattern: str,
                   output_dir: Path
                   ) -> Generator[Tuple[Path, Path], None, None]:
    for input_file in list_by_pattern(input_files, pattern):
        output_file = output_dir / (input_file.stem + '.parquet')
        yield input_file, output_file


CONVERTERS = {
    'ntuple': {
        'parquet': ntuple.to_parquet
    }
}


app = Typer()


@app.command(
    help='Convert datasets from one format to another.'
)
def convert(input_files: List[Path],
            input_format: str,
            output_format: str,
            output_dir: Path = Path('out/'),
            n_jobs: int = -1):
    output_formats_dict = CONVERTERS.get(input_format, {})
    if not output_formats_dict:
        raise ValueError(
            f"Unsupported input format: {input_format}. Supported formats: {list(CONVERTERS.keys())}."
        )

    converter_func = output_formats_dict.get(output_format, {})
    if not converter_func:
        raise ValueError(
            f"Unsupported output format for {input_format}: {output_format}. Supported formats for {input_format}: {list(output_formats_dict.keys())}."
        )

    pattern = f'*.{input_format.upper()}.root'
    files = list(file_generator(input_files, pattern, output_dir))
    iterator = tqdm(files, desc='Converting files', unit='files')
    if n_jobs != 1:
        pool = Parallel(n_jobs=n_jobs)
        pool(delayed(converter_func)(input_file, output_file)
             for input_file, output_file in iterator)
    else:
        for input_file, output_file in iterator:
            converter_func(input_file, output_file)
