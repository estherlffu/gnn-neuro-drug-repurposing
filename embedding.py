import torch
import requests

# Work around Dataverse rejecting Python's default User-Agent (copy from another file)
_original_request = requests.sessions.Session.request

def patched_request(self, method, url, **kwargs):
    headers = kwargs.get("headers") or {}
    headers.setdefault("User-Agent", "curl/8.5.0")
    kwargs["headers"] = headers
    return _original_request(self, method, url, **kwargs)

requests.sessions.Session.request = patched_request

def transe(h, r, t):
    """f_r(h, t) = ||h + r - t||_2. Smaller means a better triple."""
    return (h + r - t).norm(dim=-1)


def rotate(h, r, t):
    """RotatE distance. The embedding dimension must be even.

    Dimensions come in complex pairs. Each relation pair is normalized to
    modulus 1, then f_r(h, t) = sum_i |h_i * r_i - t_i|.
    """
    if h.shape[-1] % 2:
        raise ValueError("rotate needs an even embedding dimension")
    hr, hi = h.chunk(2, dim=-1)
    rr, ri = r.chunk(2, dim=-1)
    tr, ti = t.chunk(2, dim=-1)
    mod = torch.hypot(rr, ri).clamp_min(1e-9)
    rr, ri = rr / mod, ri / mod
    dr = hr * rr - hi * ri - tr
    di = hr * ri + hi * rr - ti
    return torch.hypot(dr, di).sum(dim=-1)


def train_knowledge_graph_embeddings(
    triples, n_entities, n_relations, d, eta, B, gamma, lam, f=transe, eps=1e-4, k_max=1000
):
    """Train knowledge graph embeddings.

    `f(h, r, t)` is a distance on embedding vectors and must broadcast over
    leading dimensions. Pass `transe`, `rotate`, or another distance.
    `triples` is an iterable of (h, r, t) integer ids.
    Returns E of shape (|E|, d) and R of shape (|R|, d).
    """
    triples = torch.as_tensor(triples, dtype=torch.long)
    bound = 6.0 / (d**0.5)
    E = torch.nn.Parameter(torch.empty(n_entities, d).uniform_(-bound, bound))
    R = torch.nn.Parameter(torch.empty(n_relations, d).uniform_(-bound, bound))

    loss_value = float("inf")
    k = 0
    entities = torch.arange(n_entities)
    while loss_value > eps and k < k_max:
        batch = triples[torch.randperm(triples.shape[0])[: min(B, triples.shape[0])]]
        h, rel, t = batch[:, 0], batch[:, 1], batch[:, 2]

        pos = f(E[h], R[rel], E[t])
        head_neg = f(E[None, :, :], R[rel][:, None, :], E[t][:, None, :])
        tail_neg = f(E[h][:, None, :], R[rel][:, None, :], E[None, :, :])
        keep_head = entities != h[:, None]
        keep_tail = entities != t[:, None]

        margin = (gamma + pos[:, None] - head_neg).clamp(min=0)[keep_head].sum()
        margin = margin + (gamma + pos[:, None] - tail_neg).clamp(min=0)[keep_tail].sum()
        reg = lam * (E[h].square().sum() + R[rel].square().sum() + E[t].square().sum())
        loss = margin + reg

        E.grad = None
        R.grad = None
        loss.backward()
        with torch.no_grad():
            if E.grad is not None:
                E -= eta * E.grad
            if R.grad is not None:
                R -= eta * R.grad
            E /= E.norm(dim=1, keepdim=True).clamp(min=1)

        loss_value = loss.item()
        k += 1

    return E.detach(), R.detach()


def load_triples(edges_path, nodes_path):
    """Read OptimusKG parquet files into trainer inputs.

    A triple is (from, relation, to). Entity ids follow the order of `id` in
    the node file, and relation ids follow the first appearance of each
    `relation` in the edge file.
    Returns a LongTensor of shape (|T|, 3), the number of entities, and the
    number of relations.
    """
    import polars as pl

    nodes = pl.read_parquet(nodes_path, columns=["id"]).select(pl.col("id").cast(pl.Utf8))
    edges = pl.read_parquet(edges_path, columns=["from", "relation", "to"]).select(
        pl.col("from").cast(pl.Utf8),
        pl.col("relation").cast(pl.Utf8),
        pl.col("to").cast(pl.Utf8),
    )
    entities = nodes.unique(maintain_order=True).with_row_index("idx")
    relations = edges.select("relation").unique(maintain_order=True).with_row_index("r")
    mapped = (
        edges.join(entities.rename({"id": "from", "idx": "h"}), on="from")
        .join(entities.rename({"id": "to", "idx": "t"}), on="to")
        .join(relations, on="relation")
    )
    if mapped.height != edges.height:
        raise ValueError("edges.parquet refers to node ids missing from nodes.parquet")

    triples = torch.tensor(mapped.select("h", "r", "t").cast(pl.Int64).to_numpy())
    return triples, entities.height, relations.height


def embed(
    edges_path, nodes_path, f=transe,
    d=8, eta=0.05, B=4, gamma=1.0, lam=1e-3,
    eps=0.0, k_max=400, max_triples=None,
):
    """Train on the parquet graph and return the embedding.

    Returns E of shape (n_entities, d) and R of shape (n_relations, d).
    `max_triples` uses only the first edges in the file.
    """
    triples, n_entities, n_relations = load_triples(edges_path, nodes_path)
    if max_triples is not None:
        triples = triples[:max_triples]
    return train_knowledge_graph_embeddings(
        triples,
        n_entities,
        n_relations,
        d, eta, B, gamma, lam,
        f=f, eps=eps, k_max=k_max,
    )


def report(name, f, triples, n_entities, n_relations, false_triples):
    E, R = train_knowledge_graph_embeddings(
        triples,
        n_entities=n_entities,
        n_relations=n_relations,
        d=8,
        eta=0.05,
        B=4,
        gamma=1.0,
        lam=1e-3,
        f=f,
        eps=0.0,
        k_max=400,
    )
    true_scores = [f(E[a], R[b], E[c]).item() for a, b, c in triples]
    false_scores = [f(E[a], R[b], E[c]).item() for a, b, c in false_triples]
    print(name, "true ", [round(s, 3) for s in true_scores])
    print(name, "false", [round(s, 3) for s in false_scores])
    print(name, "max entity norm", round(E.norm(dim=1).max().item(), 5))
    assert E.norm(dim=1).max().item() <= 1 + 1e-4
    assert sum(true_scores) < sum(false_scores)
    return E, R


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        edges_path, nodes_path = sys.argv[1], sys.argv[2]
    else:
        import optimuskg

        edges_path = optimuskg.get_file("edges.parquet")
        nodes_path = optimuskg.get_file("nodes.parquet")

    E, R = embed(edges_path, nodes_path, max_triples=4)
    print(E[0])
