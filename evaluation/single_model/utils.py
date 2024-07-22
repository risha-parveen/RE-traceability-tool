# type: ignore
import argparse
import os


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="../../data/git_data/risha-parveen/test-project", type=str, help="The input data dir")
    parser.add_argument("--model_path", default="../../model_files/online_single_model", help="The model to evaluate")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--output_dir", default="./result", help="directory to store the results")
    parser.add_argument("--overwrite", default=True, action="store_true", help="overwrite the cached data")
    parser.add_argument("--code_bert", default="microsoft/codebert-base", help="the base bert")
    parser.add_argument("--exp_name", default="test-project eval",help="id for this run of experiment")
    parser.add_argument("--chunk_query_num", default=-1, type=int,
                        help="The number of queries in each chunk of retrivial task")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    return args
