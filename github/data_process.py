import pandas as pd
from tqdm import tqdm
import logging
import configparser
import re
import sys
import numpy as np

sys.path.append("..")
sys.path.append("../..")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# if __name__ == "__main__":
#     logging.basicConfig()
#     logger = logging.getLogger()
#     logger.setLevel("INFO")

#     repo_path = 'risha-parveen/testing'

#     congig = configparser.ConfigParser()
#     config.read('credentials.cfg')

#     output_dir = './data/git_data'

#     proj_data_dir = os.path.join(output_dir, repo_path)

    