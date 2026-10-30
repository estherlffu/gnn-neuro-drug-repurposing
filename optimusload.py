import requests

# Work around Dataverse rejecting Python's default User-Agent
_original_request = requests.sessions.Session.request

def patched_request(self, method, url, **kwargs):
    headers = kwargs.get("headers") or {}
    headers.setdefault("User-Agent", "curl/8.5.0")
    kwargs["headers"] = headers
    return _original_request(self, method, url, **kwargs)

requests.sessions.Session.request = patched_request

import optimuskg

# Download and load the graph as Polars DataFrames
nodes, edges = optimuskg.load_graph(lcc=True)

# Inspect the data
print("Nodes:")
print(nodes.head())

print("\nEdges:")
print(edges.head())

print("\nNumber of nodes:", nodes.height)
print("Number of edges:", edges.height)
