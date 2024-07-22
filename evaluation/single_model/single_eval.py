# type: ignore
import logging
import os
import sys
import time
from tqdm import tqdm
import pandas as pd
import json
import math

import torch
from transformers import BertConfig

from torch.utils.data import DataLoader

# from metrices import metrics
# from utils import results_to_df, format_batch_input_for_single_bert

from utils import get_eval_args
from models import TBertS

# TODO: find a simpler way of importing files here using relative path
sys.path.insert(0, '/mnt/c/Users/gpripa/Desktop/RE-traceability-tool/github')

from git_repo_collector import Issues, Commits

class Test:
    def find_link(self, iss_id, cm_id, links):
        if str(iss_id) in links:
            if cm_id in set(links[str(iss_id)]):
                return 1
        return 0
    
    def result_to_df(self, res):
        df = pd.DataFrame()
        df['issue_id'] = [x[0] for x in res]
        df['commit_id'] = [x[1] for x in res]
        df['prediction'] = [x[2] for x in res]
        df['label'] = [x[3] for x in res]
        return df
    
    def read_artifacts(self, file_path, type):
        issues = Issues()
        commits = Commits()
        if type == 'link':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            df = pd.read_csv(file_path, keep_default_na=False)
            if type == 'commit':
                for index, row in df.iterrows():
                    commits.add_commit(commit_id=row['commit_id'], summary=row['summary'], diffs=row['diff'], files=row['files'], commit_time=row['commit_time'])
                return commits.get_commit_texts()
            else:
                for index, row in df.iterrows():
                    issues.add_issue(number=row['issue_id'], body=row['issue_desc'], comments=row['issue_comments'], createdAt=row['created_at'], updatedAt=row['closed_at'])
                return issues.get_issue_texts()

    def get_chunked_retrival_examples(self, args):
        commit_file = os.path.join(args.data_dir, "commit.csv")
        issue_file = os.path.join(args.data_dir, "issue.csv")
        link_file = os.path.join(args.data_dir, "link.json")

        self.issues_text_map = self.read_artifacts(issue_file, type="issue")
        self.commits_text_map = self.read_artifacts(commit_file, type="commit")
        links = self.read_artifacts(link_file, type="link")

        issue_id_list = self.issues_text_map.keys()
        commit_id_list = self.commits_text_map.keys()

        examples = []
        for iss_id in issue_id_list:
            for cm_id in commit_id_list:
                label = self.find_link(iss_id, cm_id, links)
                examples.append((iss_id, cm_id, label))
        return examples

    def format_batch_input(self, batch, model):
        tokenizer = model.tokenizer
        iss_ids, cm_ids = batch[0].tolist(), batch[1]
        input_ids = []
        att_masks = []
        tk_types = []
        for iss_id, cm_id in zip(iss_ids, cm_ids):
            iss_text = self.issues_text_map[iss_id]
            cm_text = self.commits_text_map[cm_id]

            feature = tokenizer.encode_plus(
                text=iss_text,
                text_pair=cm_text,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                max_length=512,
                add_special_tokens=True
            )
            input_ids.append(torch.tensor(feature["input_ids"], dtype=torch.long))
            att_masks.append(torch.tensor(feature["attention_mask"], dtype=torch.long))
            tk_types.append(torch.tensor(feature["token_type_ids"], dtype=torch.long))
        input_tensor = torch.stack(input_ids)
        att_tensor = torch.stack(att_masks)
        tk_type_tensor = torch.stack(tk_types)
        features = [input_tensor, att_tensor, tk_type_tensor]
        features = [t.to(model.device) for t in features]
        inputs = {
            'input_ids': features[0],
            'attention_mask': features[1],
            'token_type_ids': features[2],
        }
        return inputs

    def test(self, args, model, res_file_path):
        # get (issue_id, commit_id, label) array and store in retrival_examples
        chunked_examples = self.get_chunked_retrival_examples(args)
        retrival_dataloader = DataLoader(chunked_examples, batch_size = args.per_gpu_eval_batch_size)

        res = []
        for batch in tqdm(retrival_dataloader, desc="retrival evaluation"):
            iss_ids = batch[0]
            cm_ids = batch[1]
            labels = batch[2]
            with torch.no_grad():
                model.eval()
                inputs = self.format_batch_input(batch, model)
                sim_score = model.get_sim_score(**inputs)
                for n, p, prd, lb in zip(iss_ids.tolist(), cm_ids, sim_score, labels.tolist()):
                    res.append((n, p, prd, lb))
        df = self.result_to_df(res)        
        df.to_csv(res_file_path, index=False)
        return df

if __name__ == "__main__":
    args = get_eval_args()
    logging.basicConfig(level='INFO')
    logger = logging.getLogger(__name__)

    device = torch.device("cpu")
    res_file = os.path.join(args.output_dir, "raw_res.csv")

    if os.path.isfile(res_file) and not args.overwrite:
        logger.info('Evaluation result already exists')
        result_df = pd.read_csv(res_file)
    else:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.isdir('./cache'):
            os.makedirs('./cache')

        model_cache_file = os.path.join('./cache/', 'single_model_cache.pt')
        if os.path.isfile(model_cache_file):
            model = torch.load(model_cache_file)
        else:
            model = TBertS(BertConfig(), args.code_bert)
            torch.save(model, model_cache_file)

        if args.model_path and os.path.exists(args.model_path):
            model_path = os.path.join(args.model_path, 't_bert.pt')
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        else:
            raise Exception("evaluation model not found")
        logger.info("model loaded")

        start_time = time.time()
        test_dir = args.data_dir
        test = Test()
        result_df = test.test(args, model, res_file)
        exe_time = time.time() - start_time
        print(exe_time)
    