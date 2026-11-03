import requests
 
_original_request = requests.Session.request
 
def _request_with_user_agent(self, method, url, **kwargs):
    headers = kwargs.setdefault("headers", {})
    headers.setdefault("User-Agent", "curl/8.5.0")
    return _original_request(self, method, url, **kwargs)
 
requests.Session.request = _request_with_user_agent
 
import optimuskg
 
nodes, edges = optimuskg.load_graph(lcc=False)
 
print(f"{nodes.height:,} nodes, {edges.height:,} edges")
print(f"Cached files are in: {optimuskg.get_cache_dir()}")