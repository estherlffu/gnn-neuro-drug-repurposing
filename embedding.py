import torch


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


if __name__ == "__main__":
    h = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(transe(h, torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]])), torch.zeros(1))
    # (1 + 0i) * (0 + 1i) = 0 + 1i
    assert torch.allclose(rotate(h, torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 1.0]])), torch.zeros(1))

    torch.manual_seed(0)
    triples = [(0, 0, 1), (1, 0, 2), (2, 0, 3), (0, 1, 2)]
    false_triples = [(0, 0, 2), (0, 0, 3), (1, 0, 3), (3, 1, 0)]

    def report(name, f):
        E, R = train_knowledge_graph_embeddings(
            triples,
            n_entities=4,
            n_relations=2,
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

    report("transe", transe)
    report("rotate", rotate)
