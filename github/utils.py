import os
import json
import re

class DisjointSetUnion:
    """
    A class for disjoint set union data structure.

    This class provides methods for adding elements, finding the representative of a set, and merging two sets.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

def merge_sets(sets):
    """
    Merges a list of sets into disjoint sets.

    Args:
        sets (list): A list of sets.

    Returns:
        list: A list of disjoint sets.
    """
    dsu = DisjointSetUnion()
    # Add all elements to DSU
    for s in sets:
        for element in s:
            dsu.add(element)

    # Union elements within the same set
    for s in sets:
        elements = list(s)
        for i in range(1, len(elements)):
            dsu.union(elements[0], elements[i])

    # Collect all unique sets
    groups = {}
    for element in dsu.parent:
        root = dsu.find(element)
        if root not in groups:
            groups[root] = set()
        groups[root].add(element)

    return list(groups.values())

def comment_exist(issue_id, referenced_id, typename, collection):
    """
    Checks if a comment exists in an issue or pull request.

    Args:
        issue_id (int): The ID of the issue.
        referenced_id (int): The ID of the referenced issue or pull request.
        typename (str): The type of the referenced item ('Issue' or 'PullRequest').
        collection (object): The collection of issues or pull requests.

    Returns:
        bool: True if the comment exists, False otherwise.
    """
    if typename == 'Issue':
        item = collection.get_issue_by_id(referenced_id)
        if not item:
            return False
        content = item["issue_desc"] + " " + item["issue_comments"]
    else:
        item = collection.get_pr_by_id(referenced_id)
        if not item:
            return False
        content = item["body"] + " " + item["pr_comments"]
        
    pattern = rf"(?<!\w)#{issue_id}(?!\d)"
    match = re.search(pattern, content)

    return True if match else False

def chain_related_issues(chained_issue_sets, t1_link_collection, t2_link_collection, t3_link_collection):
    """
    Chains related issues together.

    Args:
        chained_issue_sets (list): A list of sets of related issues.
        link_collection (object): The collection of links.

    Returns:
        None. The results are stored in the link collection.
    """
    chained_issue_sets = merge_sets(chained_issue_sets)
    for chain in chained_issue_sets: 
        common_set = set()          
        for current_issue in chain:
            common_set.update(t1_link_collection.get_link_by_id(current_issue))
            common_set.update(t2_link_collection.get_link_by_id(current_issue))
        for current_issue in chain:
            commit_set = common_set
            if len(common_set):
                linked_commits1 = t1_link_collection.get_link_by_id(current_issue)
                linked_commits2 = t2_link_collection.get_link_by_id(current_issue)
                commit_set = commit_set - linked_commits1 - linked_commits2
                # commit_set.difference_update(linked_commits1, linked_commits2)
                if len(commit_set):
                    t3_link_collection.add_links(current_issue, commit_set)

def save_cache(data, file_name, cache_dir):
    """
    Saves data to a cache file.

    Args:
        data (object): The data to be cached.
        file_name (str): The name of the cache file.
        cache_dir (str): The directory where the cache file will be stored.

    Returns:
        None. The data is saved to the cache file.
    """
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    with open(os.path.join(cache_dir, file_name), 'w') as f:
        json.dump(data, f, indent=2)

def load_cache(file_name, cache_dir):
    """
    Loads data from a cache file.

    Args:
        file_name (str): The name of the cache file.
        cache_dir (str): The directory where the cache file is stored.

    Returns:
        object: The cached data, or None if the cache file does not exist or cannot be decoded.
    """
    try:
        with open(os.path.join(cache_dir, file_name), 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    
