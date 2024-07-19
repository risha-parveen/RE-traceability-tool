import logging
import os
import sys
import time
import tqdm

import torch
from transformers import BertConfig

from torch.utils.data import DataLoader

sys.path.append('..')
sys.path.append('../../')

# from metrices import metrics
# from utils import results_to_df, format_batch_input_for_single_bert

from utils import get_eval_args
from models import TBertS
from github.data_process import __read_artifacts

def read_OSS_examples(data_dir):
    commit_file = os.path.join(data_dir, "commit_file")
    issue_file = os.path.join(data_dir, "issue_file")
    link_file = os.path.join(data_dir, "link_file")
    examples = []
    issues = __read_artifacts(issue_file, "issue")
    commits = __read_artifacts(commit_file, "commit")
    links = __read_artifacts(link_file, "link")
    issue_index = {x.issue_id: x for x in issues}
    commit_index = {x.commit_id: x for x in commits}
    for lk in links:
        iss = issue_index[lk[0]]
        cm = commit_index[lk[1]]
        # join the tokenized content
        iss_text = iss.desc + " " + iss.comments
        cm_text = cm.summary + " " + cm.diffs
        example = {
            "NL": iss_text,
            "PL": cm_text,
            "issue_id": iss.issue_id
        }
        examples.append(example)
    return examples

def load_examples(data_dir, model, num_limit):
    cache_dir = os.path.join(data_dir, "cache")
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)
    logger.info("Creating examples from dataset file at {}".format(data_dir))
    raw_examples = read_OSS_examples(data_dir)
    if num_limit:
        raw_examples = raw_examples[:num_limit]
    # examples = Examples(raw_examples)
    # if isinstance(model, TBertT) or isinstance(model, TBertI2) or isinstance(model, TBertI):
    #     examples.update_features(model, multiprocessing.cpu_count())
    # return examples

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
    load_examples(test_dir, model=model, num_limit=args.test_num)
    # m = test(args, model, test_examples)
    exe_time = time.time() - start_time
    # m.write_summary(exe_time)