# type: ignore
import logging
import os
import sys
import time
from tqdm import tqdm
import pandas as pd
import json

import torch
from transformers import BertConfig

from torch.utils.data import DataLoader

# from metrices import metrics
# from utils import results_to_df, format_batch_input_for_single_bert

from utils import get_eval_args
from models import TBertS
sys.path.insert(0, '/mnt/c/Users/gpripa/Desktop/RE-traceability-tool/github')

from git_repo_collector import Issues, Commits
from data_process import read_OSS_artifacts

class Test:
    def __init__(self):
        self.issues = Issues()
        self.commits = Commits()

    def find_link(self, iss_id, cm_id, links):
        if str(iss_id) in links:
            if cm_id in set(links[str(iss_id)]):
                return 1
        return 0

    def get_chunked_retrival_examples(self, args):
        commit_file = os.path.join(args.data_dir, "commit.csv")
        issue_file = os.path.join(args.data_dir, "issue.csv")
        link_file = os.path.join(args.data_dir, "link.json")
        issues = read_OSS_artifacts(issue_file, type="issue", artifact=self.issues)
        commits = read_OSS_artifacts(commit_file, type="commit", artifact=self.commits)
        links = read_OSS_artifacts(link_file, type="link")

        issue_id_list = [issue['issue_id'] for issue in issues]
        commit_id_list = [commit['commit_id'] for commit in commits]

        examples = []
        for iss_id in issue_id_list:
            for cm_id in commit_id_list:
                label = self.find_link(iss_id, cm_id, links)
                examples.append((iss_id, cm_id, label))
        return examples

    def format_batch_input(self, batch, model):
        iss_ids, cm_ids, labels = batch[0], batch[1], batch[2]
        for iss_id, cm_id in zip(iss_ids, cm_ids):
            pass

    def test(self, args, model):
        # get (issue_id, commit_id, label) array and store in retrival_examples
        chunked_examples = self.get_chunked_retrival_examples(args)
        retrival_dataloader = DataLoader(chunked_examples, batch_size = 8)

        for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
            iss_ids = batch[0]
            cm_ids = batch[1]
            labels = batch[2]
            with torch.no_grad():
                model.eval()
                inputs = self.format_batch_input(batch, model)

if __name__ == "__main__":
    args = get_eval_args()
    device = torch.device("cpu")
    res_file = os.path.join(args.output_dir, "raw_res.csv")

    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    model = TBertS(BertConfig(), args.code_bert)
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.join(args.model_path, 't_bert.pt')
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        raise Exception("evaluation model not found")
    logger.info("model loaded")

    start_time = time.time()
    test_dir = args.data_dir
    test = Test()
    m = test.test(args, model)
    exe_time = time.time() - start_time
    # m.write_summary(exe_time)