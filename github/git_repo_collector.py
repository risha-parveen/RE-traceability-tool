import calendar
import logging
import os
import time
import configparser

from github import Github, \
    RateLimitExceededException  # pip install PyGithub. Lib operates on remote github to get issues
import re
import argparse
import git as local_git  # pip install GitPython. Lib operates on local repo to get commits
import pandas as pd
from tqdm import tqdm
import json
import requests
from collections import deque
import utils

logger = logging.getLogger(__name__)


class Commit:
    def __init__(self, commit_id, summary, diffs, files, commit_time):
        self.commit_id = commit_id
        self.summary = summary
        self.diffs = diffs
        self.files = files
        self.commit_time = commit_time

    def to_dict(self):
        return {
            "commit_id": self.commit_id,
            "summary": self.summary,
            "diff": self.diffs,
            "files": self.files,
            "commit_time": self.commit_time
        }

    def __str__(self):
        return str(self.to_dict())
        
class Issues:
    def __init__(self):
        self.issue_map = {}

    def add_issue(self, number, body, createdAt, updatedAt, comments):
        """
        Add an Issue to the collection.
        """
        issue = {
            "issue_id": number,
            "issue_desc": body,
            "issue_comments": comments,
            "closed_at": updatedAt,
            "created_at": createdAt
        }
        self.issue_map[number] = issue
        return issue

    def get_issue_by_id(self, issue_number):
        """
        Retrieve an Issue from the collection by its number.
        """
        return self.issue_map.get(issue_number)

    def get_all_issues(self):
        """
        Retrieve all Issues in the collection.
        """
        return list(self.issue_map.values())
    
    def get_all_issues_map(self):
        return self.issue_map

    def __str__(self):
        return str(list(self.issue_map.values()))
    
class PullRequests:
    def __init__(self):
        self.pr_map = {}

    def add_pr(self, number, body, isCrossRepository, commitsLinked, comments):
        """
        Add a pr to the collection.
        """
        pull_request = {
            "pr_id": number,
            "body": body,
            "is_cross_repository": isCrossRepository,
            "commits_linked": commitsLinked,
            "pr_comments": comments
        }
        self.pr_map[number] = pull_request
        return pull_request

    def get_pr_by_id(self, pr_number):
        """
        Retrieve a PR from the collection by its number.
        """
        return self.pr_map.get(pr_number)

    def get_all_pr(self):
        """
        Retrieve all Issues in the collection.
        """
        return list(self.pr_map.values())
    
    def get_all_pr_map(self):
        return self.pr_map

    def __str__(self):
        return str(list(self.pr_map.values()))

class Links:
    def __init__(self):
        self.link_map = {}

    def add_links(self, issue_number, commits):   
        if issue_number in self.link_map:    
            self.link_map[issue_number].update(commits)
        else:
            self.link_map[issue_number] = set(commits)
    
    def get_link_by_id(self, issue_number):
        return self.link_map[issue_number] if issue_number in self.link_map else set()

    def get_all_links(self):
        return self.link_map

class GitRepoCollector:
    def __init__(self, token, download_path, output_dir, repo_path):
        self.token = token
        self.download_path = download_path
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.cache_dir = os.path.join('../cache/' + repo_path)
        self.issues_collection = Issues()
        self.pr_collection = PullRequests()
        self.link_collection = Links()

    def clone_project(self):
        repo_url = "https://github.com/{}.git".format(self.repo_path)
        clone_path = os.path.join(self.download_path, self.repo_path)
        if not os.path.exists(clone_path):
            logger.info("Clone {}...".format(self.repo_path))
            local_git.Repo.clone_from(repo_url, clone_path)
            logger.info("finished cloning project")
        else:
            logger.info("Skip clone project as it already exist...")
        local_repo = local_git.Repo(clone_path)
        return local_repo

    def get_commits(self, commit_file_path):
        EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        local_repo = self.clone_project()
        if os.path.isfile(commit_file_path):
            logger.info("commits already existing, skip creating...")
            return
        logger.info("creating commit.csv...")
        commit_df = pd.DataFrame(columns=["commit_id", "summary", "diff", "files", "commit_time"])
        for i, commit in tqdm(enumerate(local_repo.iter_commits())):
            id = commit.hexsha
            summary = commit.summary
            create_time = commit.committed_datetime
            parent = commit.parents[0] if commit.parents else EMPTY_TREE_SHA
            differs = set()
            for diff in commit.diff(parent, create_patch=True):
                diff_lines = str(diff).split("\n")
                for diff_line in diff_lines:
                    if diff_line.startswith("+") or diff_line.startswith("-") and '@' not in diff_line:
                        differs.add(diff_line)
            files = list(commit.stats.files)
            commit = Commit(id, summary, differs, files, create_time)
            commit_df = commit_df.append(commit.to_dict(), ignore_index=True)
        commit_df.to_csv(commit_file_path)

    def store_pull_requests(self, all_pull_requests):
        for pr_edge in all_pull_requests:
            pr_node = pr_edge["node"]
            commitsLinked = []

            # if the PR is merged and has a merge commit store that commit id
            if pr_node["mergeCommit"] and pr_node["mergeCommit"]["oid"]:
                commitsLinked.append(pr_node["mergeCommit"]["oid"])

            # go through timeline items to see if there is any other commits linked 
            # through referenced event or closed event. Save all the linked commit ids

            for timeline_edge in pr_node["timelineItems"]["edges"]:
                timeline_node = timeline_edge["node"]
                if timeline_node["__typename"] == "ReferencedEvent":
                    commitsLinked.append(timeline_node["commit"]["oid"])
                if timeline_node["__typename"] == "ClosedEvent" and timeline_node["closer"]:
                    commitsLinked.append(timeline_node["closer"]["oid"])
            
            comments = pr_node["title"]
            for comment in pr_node["comments"]["edges"]:
                comments = comments + " " + comment["node"]["bodyText"]
            
            self.pr_collection.add_pr(
                number = pr_node["number"],
                body = pr_node["body"],
                isCrossRepository = pr_node["isCrossRepository"],
                commitsLinked=commitsLinked,
                comments = comments
            )

    def store_issues(self, all_issues, issue_file_path):
        issue_df = pd.DataFrame(columns=["issue_id", "issue_desc", "issue_comments", "closed_at", "created_at"])
        for issue_edge in all_issues:
            issue_node = issue_edge["node"]

            # issue title is considered as a part of comments
            # in the original TraceBERT implementation
            comments = issue_node["title"]
            for comment in issue_node["comments"]["edges"]:
                comments = comments + " " + comment["node"]["bodyText"]

            issue = self.issues_collection.add_issue(
                number=issue_node["number"],
                body=issue_node["body"],
                createdAt=issue_node["createdAt"],
                updatedAt=issue_node["updatedAt"],
                comments=comments
            )
            issue_df = issue_df.append(issue, ignore_index=True)
            issue_df.to_csv(issue_file_path)
    
    def store_links(self, all_issue_links, link_file_path):
        chained_issue_sets = []
        
        for edge in all_issue_links:
            # Iterating through each issue
            current_issue_id = edge["node"]["number"]
            commits_for_issue = []
            current_chained_issues = []
            chain_disconnected = []
            for link in edge["node"]["timelineItems"]["edges"]:
                # Iterating through the links of each issue
                link_node = link["node"]
                link_type = link_node["__typename"]

                # Referenced Event
                if link_type == "ReferencedEvent":
                    commits_for_issue.append(link_node["commit"]["oid"])

                # Closed Event
                elif link_type == "ClosedEvent":
                    closer = link_node["closer"]
                    if closer:
                        # Closed by a direct commit
                        if closer["__typename"] == "Commit":
                            commits_for_issue.append(closer["oid"])
                        # Closed by a pull request
                        elif closer["__typename"] == "PullRequest":
                            if closer["isCrossRepository"]:
                                continue
                            commits_linked = self.pr_collection.get_pr_by_id(closer["number"])["commits_linked"]
                            commits_for_issue.extend(commits_linked)

                # Connected event
                elif link_type == "ConnectedEvent":
                    subject = link_node["subject"]
                    # Issue connected manually to a pull request 
                    if subject["__typename"] == "PullRequest":  
                        if subject["isCrossRepository"]:
                            continue
                        commits_linked = self.pr_collection.get_pr_by_id(subject["number"])["commits_linked"]
                        commits_for_issue.extend(commits_linked)
                    elif subject["__typename"] == "Issue":
                        issue2_id = subject["number"]
                        current_chained_issues.extend([current_issue_id, issue2_id])
                
                # Disconnected event
                elif link_type == "DisconnectedEvent":
                    # to disconnect (remove) the links which were already store using connected event. 
                    # First connected and was later disconnected by the user. So now the link does not exist.
                    disconnected_subject = link_node["subject"]
                    if disconnected_subject["__typename"] == "PullRequest":
                        if disconnected_subject["isCrossRepository"]:
                            continue
                        commits_linked = self.pr_collection.get_pr_by_id(disconnected_subject["number"])["commits_linked"]
                        for commit in commits_linked:
                            commits_for_issue.remove(commit)
                    elif disconnected_subject["__typename"] == "Issue":
                        issue2_id = disconnected_subject["number"]
                        chain_disconnected.append(issue2_id)

                # Cross referenced event
                elif link_type == "CrossReferencedEvent":
                    source = link_node["source"]
                    typename = source["__typename"]
                    if not utils.comment_exist(
                        current_issue_id, 
                        source["number"], 
                        typename, 
                        self.issues_collection if typename=='Issue' else self.pr_collection
                    ):
                        continue
                    if typename == "PullRequest":
                        if source["isCrossRepository"]:
                            continue
                        commits_linked = self.pr_collection.get_pr_by_id(source["number"])["commits_linked"]
                        commits_for_issue.extend(commits_linked)
                    elif typename == "Issue":
                        issue2_id = source["number"]
                        current_chained_issues.extend([current_issue_id, issue2_id])
            
            if len(current_chained_issues):
                for disconnected_issue in chain_disconnected:
                    current_chained_issues.remove(disconnected_issue)
                chained_issue_sets.append(set(current_chained_issues))

            if len(commits_for_issue):
                self.link_collection.add_links(current_issue_id, commits_for_issue)
        
        utils.chain_related_issues(chained_issue_sets, self.link_collection)

        json_serializable_data = {key: list(value) for key, value in self.link_collection.get_all_links().items()}        
        with open(link_file_path, 'w') as json_file:
            json.dump(json_serializable_data, json_file, indent=2)

    def get_issue_links(self, issue_file_path, link_file_path, cache_duration=36000):
        """
        using the github graphql api collects all the basic issue details, pr details
        and also the link details between issues and commits

        arguments: issue_file_path - path to store the issue details
                    pr_file_path - path to store the pull request details
                    link_file_path - path to store the links
                    cache_duration - duration for which the cache is valid (in seconds)
        """
        from github_graphql_query import run_graphql_query

        cache_file = 'graphql_query_response.json'
        cache_data = utils.load_cache(self.cache_dir, cache_file)

        # If cache is valid, use cached data
        if cache_data and time.time() - os.path.getmtime(os.path.join(self.cache_dir, cache_file)) < cache_duration:
            all_issues, all_pull_requests, all_issue_links = cache_data
        else:
            # get all the issues, pull requests and issue links using the graphql api
            all_issues, all_pull_requests, all_issue_links = run_graphql_query(self.repo_path, self.token)
            utils.save_cache([all_issues, all_pull_requests, all_issue_links], cache_file, self.cache_dir)

        self.store_issues(all_issues, issue_file_path)
        self.store_pull_requests(all_pull_requests)
        self.store_links(all_issue_links, link_file_path)

    def create_issue_commit_dataset(self):
        output_dir = os.path.join(self.output_dir, self.repo_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        issue_file_path = os.path.join(output_dir, "issue.csv")
        commit_file_path = os.path.join(output_dir, "commit.csv")
        link_file_path = os.path.join(output_dir, "link.json")

        # first get all the commits possible using local git.
        if not os.path.isfile(commit_file_path):
            self.get_commits(commit_file_path)

        # handle the issues and pull requests
        self.get_issue_links(issue_file_path, link_file_path)
        
        return output_dir

if __name__ == "__main__":
    download_dir = 'G:/Document/git_projects'
    repo_path = 'risha-parveen/testing'
    logger.info("Processing repo: {}".format(repo_path))

    config = configparser.ConfigParser()
    config.read('../credentials.cfg')
    git_token = config['GIT']['TOKEN']

    output_dir = '../data/git_data'
    rpc = GitRepoCollector(git_token, download_dir, output_dir, repo_path)
    rpc.create_issue_commit_dataset()