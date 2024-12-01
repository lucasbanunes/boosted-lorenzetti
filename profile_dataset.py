import argparse
from lzt_utils.dataset import LztDataset
from lzt_utils.constants import N_RINGS
from lzt_utils.root import rdf_column_names, rdf_to_pandas, open_vector
from lzt_utils import formulas
import lzt_utils.plotting.pyplot as lplt
import mplhep
import matplotlib.pyplot as plt
import os
from pathlib import Path
import ROOT
import numpy as np
import pandas as pd
import json
from pathlib import Path
ROOT.EnableImplicitMT()
plt.style.use(mplhep.style.ROOT)

class ProfileDatasetConfig:

    @classmethod
    def from_json(cls, filename):
        filename = Path(filename)
        with filename.open('r') as f:
            return cls(**json.load(f))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file', required=True,
        type=str, help='Path to the json configuration file')
    return parser.parse_args()

if __name__ == '__main__':