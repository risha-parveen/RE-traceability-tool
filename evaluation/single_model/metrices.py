# type: ignore
import os
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from utils import get_eval_args

class Metrices:
    def __init__(self, data_frame, output_dir=None):
        print(data_frame, output_dir)

if __name__=="__main__":
    args = get_eval_args()
    res_file = os.path.join(args.output_dir, "raw_res.csv")
    if os.path.isfile(res_file):
        result_df = pd.read_csv(res_file)
        metrices = Metrices(data_frame=result_df, output_dir=args.output_dir)

