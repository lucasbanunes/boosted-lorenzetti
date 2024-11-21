import argparse
from typing import List
import subprocess
from lzt_utils.utils import open_directories
from tqdm import tqdm
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser(description='Merge HIT files')
    parser.add_argument(
        '-i', '--input-dir',
        action='store', dest='input_dir', required=True,
        nargs='+'
    )
    parser.add_argument(
        '-p', '--pileup-file',
        action='store', dest='pileup_file', required=True
    )
    parser.add_argument(
        '-o', '--output-dir',
        action='store', dest='output_dir', required=True
    )
    parser.add_argument(
        '--output-name',
        action='store', dest='output_name', required=True,
    )
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.pileup_file = Path(args.pileup_file)
    return args.__dict__


def main(input_dir: List[str],
         pileup_file: str,
         output_dir: str,
         output_name: str):

    filenames = list(open_directories(input_dir, 'root'))
    filenames.sort()
    for filename in tqdm(filenames):
        splitted = filename.split('.')
        splitted[0] = output_name
        out_name = '.'.join(splitted)
        filename = Path(filename)
        output_filename = output_dir / out_name
        subprocess.run(
            [
                'merge_trf.py',
                '-i', str(filename),
                '-p', str(pileup_file),
                '-o', str(output_filename)
            ]
        )


if __name__ == '__main__':
    args = parse_args()
    main(**args)
