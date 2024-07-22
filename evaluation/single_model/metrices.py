# type: ignore
import os
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from utils import get_eval_args

class Metrices:
    def __init__(self, args, df):
        self.data_frame = df
        self.output_dir = args.output_dir
        self.data_dir = args.data_dir
        self.iss_ids, self.cm_ids, self.pred, self.label = df['issue_id'], df['commit_id'], df['prediction'], df['label']
        self.group_sort = None
        self.confusion_metrices = {'tp':[], 'fp':[], 'tn':[], 'fn':[]}

    def f1_score(self, precision, recall):
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    def f2_score(self, precision, recall):
        return 5 * precision * recall / (4 * precision + recall) if precision + recall > 0 else 0

    def sort_df(self):
        issue_df = pd.read_csv(os.path.join(self.data_dir, 'clean_issue.csv'))
        commit_df = pd.read_csv(os.path.join(self.data_dir, 'clean_commit.csv'))

        confusion_directory = os.path.join(self.output_dir, 'confusion')
        if not os.path.exists(confusion_directory):
            os.makedirs(confusion_directory)

        for item in self.confusion_metrices:
            df = pd.DataFrame(self.confusion_metrices[item], columns=['issue_id', 'commit_id', 'prediction', 'label'])
            merged_df = pd.merge(df, issue_df, on='issue_id', how='left')
            merged_df = pd.merge(merged_df, commit_df, on='commit_id', how='left')
            final_df = merged_df[['issue_id', 'commit_id', 'prediction', 'label', 'issue_desc', 'issue_comments', 'summary', 'diff']]
            final_df.to_csv(os.path.join(confusion_directory, item + '.csv'), index=False)

    def f1_details(self, threshold):
        "Return true positive (tp), fp, tn,fn "
        
        # merged_df = pd.merge(result_df, issue_df, on='issue_id', how='left')

        # Merge the resulting DataFrame with commit_df on 'commit_id'
        # merged_df = pd.merge(merged_df, commit_df, on='commit_id', how='left')
        # print(merged_df)

        tp, fp, tn, fn = 0, 0, 0, 0
        for iss_id, cm_id, pred, label in zip(self.iss_ids, self.cm_ids, self.pred, self.label):
            if pred > threshold:
                pred = 1
            else:
                pred = 0
            if pred == label:
                if label == 1:
                    tp += 1
                    self.confusion_metrices['tp'].append((iss_id, cm_id, pred, label))
                else:
                    tn += 1
                    self.confusion_metrices['tn'].append((iss_id, cm_id, pred, label))
            else:
                if label == 1:
                    fn += 1
                    self.confusion_metrices['fn'].append((iss_id, cm_id, pred, label))
                else:
                    fp += 1
                    self.confusion_metrices['fp'].append((iss_id, cm_id, pred, label))
        self.sort_df()
        return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    
    def get_precision_recall_curve(self, fig_name):
        precision, recall, thresholds = precision_recall_curve(self.label, self.pred)
        max_f1 = 0
        max_f2 = 0
        max_threshold = 0
        for p, r, tr in zip(precision, recall, thresholds):
            f1 = self.f1_score(p, r)
            f2 = self.f2_score(p, r)
            if f1 >= max_f1:
                max_f1 = f1
                max_threshold = tr
            if f2 >= max_f2:
                max_f2 = f2
        viz = PrecisionRecallDisplay(
            precision=precision, recall=recall)
        viz.plot()
        if os.path.isdir(self.output_dir):
            fig_path = os.path.join(self.output_dir, fig_name)
            plt.savefig(fig_path)
            plt.close()
        detail = self.f1_details(max_threshold)
        return round(max_f1, 3), round(max_f2, 3), detail, max_threshold
    
    def precision_at_K(self, k=1):
        if self.group_sort is None:
            self.group_sort = self.data_frame.groupby(["issue_id"]).apply(
                lambda x: x.sort_values(["prediction"], ascending=False)).reset_index(drop=True)
        group_tops = self.group_sort.groupby('issue_id')
        cnt = 0
        hits = 0
        for iss_id, group in group_tops:
            for index, row in group.head(k).iterrows():
                hits += 1 if row['label'] == 1 else 0
            cnt += 1
        return round(hits / cnt if cnt > 0 else 0, 3)
    
    def MAP_at_K(self, k=1):
        if self.group_sort is None:
            self.group_sort = self.data_frame.groupby(["issue_id"]).apply(
                lambda x: x.sort_values(["prediction"], ascending=False)).reset_index(drop=True)
        group_tops = self.group_sort.groupby('issue_id')
        ap_sum = 0
        for iss_id, group in group_tops:
            group_hits = 0
            ap = 0
            for i, (index, row) in enumerate(group.head(k).iterrows()):
                if row['label'] == 1:
                    group_hits += 1
                    ap += group_hits / (i + 1)
            ap = ap / group_hits if group_hits > 0 else 0
            ap_sum += ap
        map = ap_sum / len(group_tops) if len(group_tops) > 0 else 0
        return round(map, 3)
    
    def MRR(self):
        if self.group_sort is None:
            self.group_sort = self.data_frame.groupby(["issue_id"]).apply(
                lambda x: x.sort_values(["prediction"], ascending=False)).reset_index(drop=True)
        group_tops = self.group_sort.groupby('issue_id')
        mrr_sum = 0
        for iss_id, group in group_tops:
            rank = 0
            for i, (index, row) in enumerate(group.iterrows()):
                rank += 1
                if row['label'] == 1:
                    mrr_sum += 1.0 / rank
                    break
        return mrr_sum / len(group_tops)
    
    def get_all_metrices(self):
        pk3 = self.precision_at_K(3)
        pk2 = self.precision_at_K(2)
        pk1 = self.precision_at_K(1)
        
        best_f1, best_f2, details, f1_threshold = self.get_precision_recall_curve("pr_curve.png")
        map = self.MAP_at_K(3)
        print(map)
        mrr = self.MRR()
        print(mrr)
        return {
            'pk3': pk3,
            'pk2': pk2,
            'pk1': pk1,
            'f1': best_f1,
            'f2': best_f2,
            'map': map,
            'mrr': mrr,
            'details': details,
            'f1_threshold': f1_threshold
        }

    def write_summary(self, exe_time):
        summary_path = os.path.join(self.output_dir, "summary.txt")
        res = self.get_all_metrices()
        pk3, pk2, pk1 = res['pk3'], res['pk2'], res['pk1']
        best_f1, best_f2, details = res['f1'], res['f2'], res['details']
        map, mrr = res['map'], res['mrr']
        summary = "\npk3={}, pk2={},pk1={} best_f1 = {}, bets_f2={}, MAP={}, MRR={}, exe_time={},f1_threshold={}\n".format(
            pk3,
            pk2,
            pk1,
            best_f1,
            best_f2,
            map,
            mrr,
            exe_time,
            res['f1_threshold']
        )
        with open(summary_path, 'w') as fout:
            fout.write(summary)
            fout.write(str(details))
        print(summary)

if __name__=="__main__":
    args = get_eval_args()
    res_file = os.path.join(args.output_dir, "raw_res.csv")
    if os.path.isfile(res_file):
        result_df = pd.read_csv(res_file)
        metrices = Metrices(args, df=result_df)
        metrices.write_summary(1)

