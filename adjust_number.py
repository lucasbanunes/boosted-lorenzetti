import argparse
import os
from tqdm import tqdm
from lzt_utils.dataset import LztDataset, FILE_DIRECTORIES
import re
import logging
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Renames the root files generated with lorenzetti to end "
        "with .i.root instead of .root.i")
    parser.add_argument('--dir-paths', nargs='+',
                        help='Lorenzetti datasets path',
                        dest='dir_paths',
                        required=True)
    return parser.parse_args()


logging.basicConfig(level=logging.DEBUG)
if __name__ == '__main__':
    args = parse_args()
    file_match = re.compile(r'\.root\.[0-9]*$')
    for path in args.dir_paths:
        dataset = LztDataset(path)
        for directory in FILE_DIRECTORIES:
            dir_path = os.path.join(path, directory)
            if not os.path.exists(dir_path):
                logging.info(f'{dir_path} does not exist')
                continue
            logging.info(f'Processing {dir_path}')
            glob_path = os.path.join(dir_path, '*.root.*')
            for filename in tqdm(glob.glob(glob_path)):
                if not file_match.search(filename):
                    continue
                splitted = filename.split('.')
                aux = splitted[-1]
                splitted[-1] = splitted[-2]
                splitted[-2] = aux
                new_name = '.'.join(splitted)
                os.rename(filename, new_name)
