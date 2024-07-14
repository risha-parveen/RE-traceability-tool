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

    def __str__(self):
        return str(list(self.issue_map.values()))
    
class PullRequests:
    def __init__(self):
        self.pr_map = {}

    def add_pr(self, number, isCrossRepository, commitsLinked):
        """
        Add a pr to the collection.
        """
        pull_request = {
            "pr_id": number,
            "is_cross_repository": isCrossRepository,
            "commits_linked": commitsLinked
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

    def __str__(self):
        return str(list(self.pr_map.values()))

class GitRepoCollector:
    CACHE_DIR = './cache'

    def __init__(self, token, download_path, output_dir, repo_path):
        self.token = token
        self.download_path = download_path
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.issues_collection = Issues()
        self.pr_collection = PullRequests()

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

    def save_cache(self, file_name, data):
        if not os.path.exists(self.CACHE_DIR):
            os.makedirs(self.CACHE_DIR)
        with open(os.path.join(self.CACHE_DIR, file_name), 'w') as f:
            json.dump(data, f)

    def load_cache(self, file_name):
        try:
            with open(os.path.join(self.CACHE_DIR, file_name), 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def make_github_graphql_request(self, token, variables):
        """
        Makes a GraphQL request to GitHub API.

        Args:
        - token: GitHub personal access token
        - query: GraphQL query string
        - variables: Variables for the GraphQL query

        Returns:
        - JSON response from GitHub API
        """
        from github_graphql_query import GITHUB_GRAPHQL_QUERY, GITHUB_GRAPHQL_URL

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                GITHUB_GRAPHQL_URL,
                headers=headers,
                json={"query": GITHUB_GRAPHQL_QUERY, "variables": variables}
            )
            if response.status_code == 200:
                return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making GraphQL request: {e}")
            return None

    def run_graphql_query(self):

        owner, name = self.repo_path.split('/')

        variables = {
            "issuesTimelineCursor": None,
            "issuesCursor": None,
            "pullRequestsCursor": None,
            "owner": owner,
            "name": name
        }

        all_issues = []
        all_pull_requests = []
        all_issue_links = []

        # Make the GraphQL request using the imported function
        while True:
            hasNextPage = False
            response_data = self.make_github_graphql_request(self.token, variables)
            if response_data:
                try:

                    basic_issues = response_data["data"]["repository"]["basicIssues"]
                    pull_requests = response_data["data"]["repository"]["pullRequests"]
                    issues_with_timeline = response_data["data"]["repository"]["issuesWithTimeline"]

                    # Append issues and pull requests data
                    all_issues.extend(basic_issues["edges"])
                    all_pull_requests.extend(pull_requests["edges"])
                    all_issue_links.extend(issues_with_timeline["edges"])

                    # Check for pagination
                    if basic_issues["pageInfo"]["hasNextPage"]:
                        variables["issuesCursor"] = basic_issues["pageInfo"]["endCursor"]
                        hasNextPage = True
                    if pull_requests["pageInfo"]["hasNextPage"]:
                        variables["pullRequestsCursor"] = pull_requests["pageInfo"]["endCursor"]
                        hasNextPage = True
                    if issues_with_timeline["pageInfo"]["hasNextPage"]:
                        variables["issuesTimelineCursor"] = issues_with_timeline["pageInfo"]["endCursor"]
                        hasNextPage = True
                    
                    if not hasNextPage:
                        break
                except:
                    print('Error occured while executing query')
                    break
            else:
                print("GraphQL request failed, stopping further processing.")
                break
        return all_issues, all_pull_requests, all_issue_links
    
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
            

            self.pr_collection.add_pr(
                number = pr_node["number"],
                isCrossRepository = pr_node["isCrossRepository"],
                commitsLinked=commitsLinked
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
    
    def get_issue_links(self, issue_file_path, link_file_path, cache_duration=36000):
        """
        using the github graphql api collects all the basic issue details, pr details
        and also the link details between issues and commits

        arguments: issue_file_path - path to store the issue details
                    pr_file_path - path to store the pull request details
                    link_file_path - path to store the links
                    cache_duration - duration for which the cache is valid (in seconds)
        """

        cache_file = 'graphql_cache.json'
        cache_data = self.load_cache(cache_file)

        # If cache is valid, use cached data
        if cache_data and time.time() - os.path.getmtime(os.path.join(self.CACHE_DIR, cache_file)) < cache_duration:
            all_issues, all_pull_requests, all_issue_links = cache_data
        else:
            # get all the issues, pull requests and issue links using the graphql api
            all_issues, all_pull_requests, all_issue_links = self.run_graphql_query()
            self.save_cache(cache_file, [all_issues, all_pull_requests, all_issue_links])

        if not os.path.isfile(issue_file_path):
            self.store_issues(all_issues, issue_file_path)

            # we are not saving the pull requests in a csv for now.
        self.store_pull_requests(all_pull_requests)


    def create_issue_commit_dataset(self):
        output_dir = os.path.join(self.output_dir, self.repo_path)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        issue_file_path = os.path.join(output_dir, "issue.csv")
        commit_file_path = os.path.join(output_dir, "commit.csv")
        link_file_path = os.path.join(output_dir, "link.csv")

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
    config.read('credentials.cfg')
    git_token = config['GIT']['TOKEN']

    output_dir = './data/git_data'
    rpc = GitRepoCollector(git_token, download_dir, output_dir, repo_path)
    rpc.create_issue_commit_dataset()