import pandas as pd
from tqdm import tqdm
import logging
import configparser
import re
import sys
import numpy as np
import argparse
import os
import json

from git_repo_collector import GitRepoCollector, Commit, Issues

sys.path.append("..")
sys.path.append("../..")

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def __read_artifacts(file_path, type):
    if type == 'link':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        df = pd.read_csv(file_path)
        if type == 'commit':
            commits = []
            for index, row in df.iterrows():
                commit = Commit(commit_id=row['commit_id'], summary=row['summary'], diffs=row['diff'], files=row['files'], commit_time=row['commit_time'])
                commits.append(commit.get_commit())
            return commits 
        else:
            issues = Issues()
            for index, row in df.iterrows():
                issues.add_issue(number=row['issue_id'], body=row['issue_desc'], comments=row['issue_comments'], createdAt=row['created_at'], updatedAt=row['closed_at'])
            return issues.get_all_issues()

def read_artifacts(proj_data_dir):
    commit_file = os.path.join(proj_data_dir, "commit.csv")
    issue_file = os.path.join(proj_data_dir, "issue.csv")
    link_file = os.path.join(proj_data_dir, "link.json")
    
    issues = __read_artifacts(issue_file, type="issue")
    commits = __read_artifacts(commit_file, type="commit")
    links = __read_artifacts(link_file, type="link")

    return issues, commits, links

def clean_artifacts(proj_dir):
    issue, commit, link = read_artifacts(proj_dir)
    clean_issue_file = os.path.join(proj_dir, "clean_issue.csv")
    clean_commit_file = os.path.join(proj_dir, "clean_commit.csv")

    return None, None


if __name__ == "__main__":

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel("INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=str, help='Path of the github repo to be processed')
    parser.add_argument('--output_dir', type=str, help='Output directory path for processed data')

    default_project = 'risha-parveen/testing'
    default_output_dir = '../data/git_data'

    args = parser.parse_args()

    repo_path = args.repo if args.repo else default_project 
    output_dir = args.output_dir if args.output_dir else default_output_dir

    config = configparser.ConfigParser()
    config.read('../credentials.cfg')

    proj_data_dir = os.path.join(output_dir, repo_path)
        
    if not os.path.exists(os.path.join(proj_data_dir, 'issue.csv')):
        # if the issue_csv is not available
        logger.info("Processing repo: {}".format(repo_path))
        git_token = config['GIT']['TOKEN']
        download_dir = 'G:/Document/git_projects'
        rpc = GitRepoCollector(git_token, download_dir, output_dir, repo_path)
        rpc.create_issue_commit_dataset()

    clean_issue_file = os.path.join(proj_data_dir, 'clean_issue.csv')
    clean_commits_file = os.path.join(proj_data_dir, 'clean_commit.csv')

    # if not os.path.exists(clean_issue_file) or not os.path.exists(clean_commits_file):
    #     clean_issues, clean_commits, clean_links = clean_artifacts(proj_data_dir)

    clean_issues, clean_commits = clean_artifacts(proj_data_dir)