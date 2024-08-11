# type: ignore
from args import get_eval_args
import os
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

args = get_eval_args()

def read_json_file(file_name):
    file_path = os.path.join(args.root_data_dir, args.repo_path, file_name)
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# def find_optimal_threshold(result_df):
#     fpr, tpr, thresholds = roc_curve(result_df['label'], result_df['prediction'])
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, label='ROC Curve')
#     plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend(loc='best')
#     plt.grid(True)
#     fig_path = os.path.join(args.output_dir, 'probability')
#     plt.savefig(fig_path)
#     plt.close()

def logit_transform(p):
    return np.log(p / (1 - p))

def sort_positives(result_df):
    iss_ids, cm_ids, predictions = result_df['issue_id'], result_df['commit_id'], result_df['prediction']

    t1_link, t2_link, t3_link = read_json_file('t1.json'), read_json_file('t2.json'), read_json_file('t3.json')
    threshold = 0.00026
    
    result_df['link_type'] = 4
    result_df['label'] = 0


    for idx, (iss, cm) in enumerate(zip(iss_ids, cm_ids)):
        issue = str(iss)
        label = 0
        if issue in t1_link:
            commits = set(t1_link[issue])
            if cm in commits:
                result_df.at[idx, 'link_type'] = 1
                label = 1
                continue
        if issue in t2_link:
            commits = set(t2_link[issue])
            if cm in commits:
                result_df.at[idx, 'link_type'] = 2
                continue
        if issue in t3_link:
            commits = set(t3_link[issue])
            if cm in commits:
                result_df.at[idx, 'link_type'] = 3
        result_df.at[idx, 'label'] = label
    
    result_df = result_df.sort_values(by=['link_type', 'prediction'], ascending=[True, False], ignore_index = True)
    # positive_df = result_df[result_df['prediction'] > threshold]
    # negative_df = result_df[result_df['prediction'] <= threshold]
    # positive_df.to_csv(os.path.join(args.output_dir, 'positives.csv'), index=False)
    # negative_df.to_csv(os.path.join(args.output_dir, 'negatives.csv'), index=False)   

if __name__ == '__main__':
    res_file = os.path.join(args.output_dir, "raw_res.csv")
    if os.path.isfile(res_file):
        result_df = pd.read_csv(res_file)
        sort_positives(result_df)
