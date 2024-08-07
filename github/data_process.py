# type: ignore
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
import csv

from git_repo_collector import GitRepoCollector, Commits, Issues
import nltk

# nltk.download('punkt')
from nltk.tokenize import word_tokenize

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class DataProcess:

    def __save_artifacts(self, art_list, output_file):
        df = pd.DataFrame(art_list)
        df.to_csv(output_file, index=True)

    def read_OSS_artifacts(self, file_path, type, artifact = None):
        if type == 'link':
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            df = pd.read_csv(file_path, keep_default_na=False)
            if type == 'commit':
                for index, row in df.iterrows():
                    artifact.add_commit(commit_id=row['commit_id'], summary=row['summary'], diffs=row['diff'], files=row['files'], commit_time=row['commit_time'])
                return artifact.get_all_commits()
            else:
                for index, row in df.iterrows():
                    artifact.add_issue(number=row['issue_id'], body=row['issue_desc'], comments=row['issue_comments'], createdAt=row['created_at'], updatedAt=row['closed_at'])
                return artifact.get_all_issues()

    def read_artifacts(self, proj_data_dir):
        issues = Issues()
        commits = Commits()
        commit_file = os.path.join(proj_data_dir, "commit.csv")
        issue_file = os.path.join(proj_data_dir, "issue.csv")
        link_file = os.path.join(proj_data_dir, "link.json")
        
        issues = self.read_OSS_artifacts(issue_file, type="issue", artifact=issues)
        commits = self.read_OSS_artifacts(commit_file, type="commit", artifact=commits)
        links = self.read_OSS_artifacts(link_file, type="link")

        return issues, commits, links

    def clean_artifacts(self, proj_dir):
        issue, commit, _ = self.read_artifacts(proj_dir)
        clean_issue_file = os.path.join(proj_dir, "clean_issue.csv")
        clean_commit_file = os.path.join(proj_dir, "clean_commit.csv")

        clean_issues = dict()
        clean_commits = dict()

        if not os.path.isfile(clean_issue_file):
            for iss in tqdm(issue):
                if pd.isnull(iss["issue_desc"]):
                    iss["issue_desc"] = ""
                iss["issue_desc"] = re.sub("<!-.*->", "", iss["issue_desc"])
                iss["issue_desc"] = re.sub("```.*```", "", iss["issue_desc"], flags=re.DOTALL)
                iss["issue_desc"] = " ".join(word_tokenize(iss["issue_desc"]))
                iss["issue_comments"] = " ".join(word_tokenize(iss["issue_comments"]))  # use only the first comment (title)
        
                clean_issues[iss["issue_id"]] = iss
        else:
            tmp_issues = read_OSS_artifacts(clean_issue_file, type="issue")
            for iss in tmp_issues:
                clean_issues[iss["issue_id"]] = iss

        if not os.path.isfile(clean_commit_file):
            for cm in tqdm(commit):
                diff_sents = eval(cm["diff"])
                diff_tokens = []
                for sent in diff_sents:
                    sent = sent.strip("+- ")
                    diff_tokens.extend(word_tokenize(sent))
                cm["diff"] = " ".join(diff_tokens)
                cm["summary"] = " ".join(word_tokenize(cm["summary"]))
                clean_commits[cm["commit_id"]] = cm
        else:
            tmp_commit = self.read_OSS_artifacts(clean_commit_file, type="commit")
            for cm in tmp_commit:
                clean_commits[cm["commit_id"]] = cm

        # save clean artifacts
        self.__save_artifacts(clean_issues.values(), output_file=clean_issue_file)
        self.__save_artifacts(clean_commits.values(), output_file=clean_commit_file)

        

def main(args):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel("INFO")

    config = configparser.ConfigParser()
    config.read('../credentials.cfg')

    proj_data_dir = os.path.join(args.root_data_dir, args.repo_path)
        
    if not os.path.exists(os.path.join(proj_data_dir, 'clean_issue.csv')):
        # if the issue_csv is not available
        logger.info("Processing repo: {}".format(args.repo_path))
        print(config)
        git_token = config['GIT']['TOKEN']
        download_dir = 'G:/Document/git_projects'
        rpc = GitRepoCollector(git_token, download_dir, args.root_data_dir, args.repo_path)
        rpc.create_issue_commit_dataset()

        data_processing = DataProcess()

        data_processing.clean_artifacts(proj_data_dir)
    logger.info('Cleaned issues and commits are stored')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", default="risha-parveen/testing")
    parser.add_argument("--root_data_dir", default="../data/git_data_original")

    main(parser.parse_args())
    