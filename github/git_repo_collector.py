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
    
    def get_commit(self):
        return self.to_dict()
        
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
    """
    A class for collecting and processing data from a GitHub repository.

    This class provides methods for cloning a GitHub repository, retrieving and storing issues, pull requests, 
    and commits, and creating a dataset of issues and commits. The data is stored in specified files.

    Attributes:
        token (str): The GitHub token for authentication.
        download_path (str): The path where the GitHub repository will be cloned.
        repo_path (str): The path to the GitHub repository.
        output_dir (str): The directory where the output data files will be stored.
        cache_dir (str): The directory where the cache data will be stored.
    """

    def __init__(self, token, download_path, output_dir, repo_path, cache_dir = None):
        self.token = token
        self.download_path = download_path
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.cache_dir = os.path.join('../cache/' + repo_path) if not cache_dir else cache_dir
        self.issues_collection = Issues()
        self.pr_collection = PullRequests()
        self.link_collection = Links()

    def clone_project(self):
        """
        Clones a GitHub repository to a local directory.

        Args:
            None.

        Returns:
            local_git.Repo: The cloned local repository.
        """
        repo_url = "https://github.com/{}.git".format(self.repo_path)
        clone_path = os.path.join(self.download_path, self.repo_path)
        if not os.path.exists(clone_path):
            print("Clone {}...".format(self.repo_path))
            local_git.Repo.clone_from(repo_url, clone_path)
            print("finished cloning project")
        else:
            print("Skip clone project as it already exist...")
        local_repo = local_git.Repo(clone_path)
        return local_repo

    def get_commits(self, commit_file_path):
        """
        Retrieves commit data from a cloned local repository, processes the data, and stores it in a CSV file.
        If the CSV file already exists, the method logs a message and returns, skipping the creation process.

        Args:
            commit_file_path (str): The path where the resulting CSV file will be saved.

        Returns:
            None. The results are stored in a CSV file at the specified path.
        """
        EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
        local_repo = self.clone_project()
        if os.path.isfile(commit_file_path):
            print("commits already existing, skip creating...")
            return
        print("creating commit.csv...")
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
        """
        Store pull requests and their associated information in a collection.

        Args:
            all_pull_requests (list): A list of pull requests

        Returns:
            None. The results are stored in the 'pr_collection' attribute of the 'self' object.
        """
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
            
            # PR title and comment body text are concatenated to form a string called comments.
            comments = pr_node["title"]
            for comment in pr_node["comments"]["edges"]:
                comments = comments + " " + comment["node"]["bodyText"]
            
            # add each PR to the pr_collection
            self.pr_collection.add_pr(
                number = pr_node["number"],
                body = pr_node["body"],
                isCrossRepository = pr_node["isCrossRepository"],
                commitsLinked=commitsLinked,
                comments = comments
            )

    def store_issues(self, all_issues, issue_file_path):
        """
        Processes and stores issue data in a DataFrame, then saves it to a CSV file.

        Args:
            all_issues (list): A list of issues, each represented as a dictionary.
            issue_file_path (str): The path to the CSV file where the issue data will be stored.

        Returns:
            None. The DataFrame is saved as a CSV file at the specified path.
        """
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
        """
        Processes each issue link in the all_issue_links list, handles different types of events 
        and adds the processed links to a collection. It then saves the link data to a file at the specified path.
        
        Args:
            all_issue_links (list): A list of issue links, each represented as a dictionary.
            link_file_path (str): The path to the file where the link data will be stored.

        Returns:
            None. The results are stored in the specified file.
        """
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
                        # checking if the pull request belong to the same repository or not 
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
                    # check if the comments that is referencing the current issue is still present.
                    # to ensure that the comment linking this issue was made and was not deleted later. 
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
                        # add the issue to the current_chained_issues set if an issue is linked to an issue
                        current_chained_issues.extend([current_issue_id, issue2_id])
            
            if len(current_chained_issues):
                for disconnected_issue in chain_disconnected:
                    current_chained_issues.remove(disconnected_issue)
                chained_issue_sets.append(set(current_chained_issues))

            if len(commits_for_issue):
                # the commits that are gathered for the current issue is linked to it.
                self.link_collection.add_links(current_issue_id, commits_for_issue)
        
        utils.chain_related_issues(chained_issue_sets, self.link_collection)

        json_serializable_data = {key: list(value) for key, value in self.link_collection.get_all_links().items()}        
        with open(link_file_path, 'w') as json_file:
            json.dump(json_serializable_data, json_file, indent=2)

    def get_issue_links(self, issue_file_path, link_file_path, cache_duration=36000):
        """
        Retrieves issue links from a GitHub repository and stores them in specified files.

        Args:
            issue_file_path (str): The path to the file where issue data will be stored.
            link_file_path (str): The path to the file where link data will be stored.
            cache_duration (int, optional): The duration for which the cache is considered valid. Defaults to 36000 seconds.

        Returns:
            None. The results are stored in the specified files.
        """
        from github_graphql_query import run_graphql_query

        cache_file = 'graphql_query_response.json'
        cache_data = utils.load_cache(self.cache_dir, cache_file)

        # If cache is valid, use cached data
        try:
            if cache_data and time.time() - os.path.getmtime(os.path.join(self.cache_dir, cache_file)) < cache_duration:
                print('Cache file exist in cache directory. Fetching query response from cache')
                all_issues, all_pull_requests, all_issue_links = cache_data
            else:
                # get all the issues, pull requests and issue links using the graphql api
                all_issues, all_pull_requests, all_issue_links = run_graphql_query(self.repo_path, self.token)
                utils.save_cache([all_issues, all_pull_requests, all_issue_links], cache_file, self.cache_dir)

            self.store_issues(all_issues, issue_file_path)
            print('Stored issues in '+ issue_file_path )

            self.store_pull_requests(all_pull_requests)
            print('Stored pull requests data' )

            self.store_links(all_issue_links, link_file_path)
            print('Stored links in '+ link_file_path )
        except:
            print('Error occured')

    def create_issue_commit_dataset(self):
        """
        Creates a dataset of issues and commits from a GitHub repository and stores them in specified files.

        Args:
            None.

        Returns:
            str: The path to the directory where the issue and commit data files are stored.
        """
        output_dir = os.path.join(self.output_dir, self.repo_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        issue_file_path = os.path.join(output_dir, "issue.csv")
        commit_file_path = os.path.join(output_dir, "commit.csv")
        link_file_path = os.path.join(output_dir, "link.json")

        # Get all the commits possible using local git.        
        if not os.path.isfile(commit_file_path):
            print('Fetching commits...')
            try:
                self.get_commits(commit_file_path)
                print('Commits saved to ' + commit_file_path)
            except:
                print('Error occured in fetching commits')
        else:
            print('Commits already stored in '+ commit_file_path)

        # handle the issues and pull requests
        self.get_issue_links(issue_file_path, link_file_path)
        return output_dir

if __name__ == "__main__":
    download_dir = 'G:/Document/git_projects'
    repo_path = 'risha-parveen/testing'

    config = configparser.ConfigParser()
    config.read('../credentials.cfg')
    git_token = config['GIT']['TOKEN']

    output_dir = '../data/git_data'
    rpc = GitRepoCollector(git_token, download_dir, output_dir, repo_path)
    rpc.create_issue_commit_dataset()