import requests
# GitHub GraphQL API URL
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"

GITHUB_GRAPHQL_QUERY = """
query($issuesTimelineCursor: String, $issuesCursor: String, $pullRequestsCursor: String, $owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    issuesWithTimeline: issues(first: 50, after: $issuesTimelineCursor) {
      pageInfo {
        endCursor
        hasNextPage
      }
      edges {
        node {
          number
          timelineItems(first: 50, itemTypes: [CROSS_REFERENCED_EVENT, CONNECTED_EVENT, REFERENCED_EVENT, CLOSED_EVENT, DISCONNECTED_EVENT]) {
            edges {
              node {
                __typename
                ... on DisconnectedEvent {
                  id
                  subject {
                    __typename
                    ... on PullRequest {
                      number
                      isCrossRepository
                    }
                    ... on Issue {
                      number
                    }
                  }
                }
                ... on ReferencedEvent {
                  id
                  commit {
                    oid
                  }
                }
                ... on ClosedEvent {
                  id
                  closer {
                    __typename
                    ... on Commit {
                      oid
                    }
                    ... on PullRequest {
                      number
                      isCrossRepository
                    }
                  }
                }
                ... on CrossReferencedEvent {
                  id
                  source {
                    __typename
                    ... on PullRequest {
                      number
                      isCrossRepository
                    }
                    ... on Issue {
                      number
                    }
                  }
                }
                ... on ConnectedEvent {
                  id
                  subject {
                    __typename
                    ... on PullRequest {
                      number
                      isCrossRepository
                    }
                    ... on Issue {
                      number
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    basicIssues: issues(first: 50, after: $issuesCursor) { 
    	pageInfo {
        endCursor
        hasNextPage
      }
      edges {
        node {
          number
          title
          body
          createdAt
          updatedAt
          comments(first: 100) {
            edges {
              node {
                id
                bodyText
              }
            }
          }
        }
      }
    }
    pullRequests(first: 50, after: $pullRequestsCursor) {
      pageInfo {
        endCursor
        hasNextPage
      }
      edges {
        node {
          number
          title
          body
          isCrossRepository
          mergeCommit { 
          	oid
          }
          comments(first: 100) {
            edges {
              node {
                id
                bodyText
              }
            }
          }
          timelineItems(first:50, itemTypes: [REFERENCED_EVENT, CLOSED_EVENT]) { 
          	edges {
              node {
                __typename
                ... on ReferencedEvent {
                  id
                  commit {
                    oid
                  }
                }
                ... on ClosedEvent {
                  id
                  closer {
                    __typename
                    ... on Commit {
                      oid
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}

"""

def make_github_graphql_request(token, variables):
        """
        Makes a GraphQL request to GitHub API.

        Args:
        - token: GitHub personal access token
        - query: GraphQL query string
        - variables: Variables for the GraphQL query

        Returns:
        - JSON response from GitHub API
        """

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
        

def run_graphql_query(repo_path, token):

        owner, name = repo_path.split('/')

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
            response_data = make_github_graphql_request(token, variables)
            
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