
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
          isCrossRepository
          mergeCommit { 
          	oid
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


# # Function to make GraphQL request
# def make_github_graphql_request(token, variables):
#     """
#     Makes a GraphQL request to GitHub API.

#     Args:
#     - token: GitHub personal access token
#     - query: GraphQL query string
#     - variables: Variables for the GraphQL query

#     Returns:
#     - JSON response from GitHub API
#     """
#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json"
#     }

#     try:
#         response = requests.post(
#             GITHUB_GRAPHQL_URL,
#             headers=headers,
#             json={"query": GITHUB_GRAPHQL_QUERY, "variables": variables}
#         )
#         if response.status_code == 200:
#             result = response.json()

#     except requests.exceptions.RequestException as e:
#         print(f"Error making GraphQL request: {e}")
#         return None
