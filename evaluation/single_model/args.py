# type: ignore
import argparse
import os


def get_eval_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--repo_path", default="risha-parveen/test-project", help="OSS repository to be evaluated")
 
    parser.add_argument("--root_data_dir", default="../../data/git_data", type=str, help="The input data dir for evaluation")
    
    parser.add_argument("--model_path", default="../../model_files/checkpoint-3001", help="The model to evaluate")
   
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    
    parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    
    parser.add_argument("--output_dir", default="./result_lora", help="directory to store the results")
    
    parser.add_argument("--overwrite", default=False, action="store_true", help="overwrite the cached data")
    
    parser.add_argument("--code_bert", default="microsoft/codebert-base", help="the base bert")
    
    parser.add_argument("--exp_name", default="test-project",help="id for this run of experiment")
    
    parser.add_argument("--chunk_query_num", default=-1, type=int,
                        help="The number of queries in each chunk of retrivial task")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    return args
