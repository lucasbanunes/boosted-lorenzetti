import argparse
import os
from tqdm import tqdm
from lzt_utils.dataset import LztDataset, FILE_DIRECTORIES
import logging
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description="Renames the root files generated with lorenzetti "
        "to match their directory names")
    parser.add_argument('--dir-paths', nargs='+',
                        help='Lorenzetti datasets path',
                        dest='dir_paths',
                        required=True)
    return parser.parse_args()


logging.basicConfig(level=logging.DEBUG)
if __name__ == '__main__':
    args = parse_args()
    for path in args.dir_paths:
        dataset = LztDataset(path)
        for directory in FILE_DIRECTORIES:
            dir_path = os.path.join(path, directory)
            if not os.path.exists(dir_path):
                logging.info(f'{dir_path} does not exist')
                continue
            logging.info(f'Processing {dir_path}')
            glob_path = os.path.join(dir_path, '*.root')
            for filename in glob.glob(glob_path):
                name = os.path.basename(filename)
                splitted = name.split('.')
                splitted[0] = 'gg2H2ZZ2ee'
                new_name = '.'.join(splitted)
                new_filename = os.path.join(dir_path, new_name)
                logging.info(f'Renaming {name} to {new_name}')
                os.rename(filename, new_filename)
