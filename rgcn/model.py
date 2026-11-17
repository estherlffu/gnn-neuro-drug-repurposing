"""R-GCN encoder and drug–disease scoring.

The previous trainer sampled a 10-then-5 neighborhood out of a ~22M-edge graph
and spent almost all of its loss on relations other than DRG-DIS|INDICATION.
Indication edges are about 0.3% of OptimusKG, so that model ranks at chance
(AUPRC below 0.1).

This encoder runs on the whole message-passing graph. Drug–disease edges are
not in that graph: the indication has to come from drug–gene–disease (and
phenotype / pathway) paths. Relation weights start as the identity, and the
score reads the 1-hop layer first, so two nodes that share a gene are already
closer than a random pair before any training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelConv(nn.Module):
    """Relational graph convolution with per-relation mean aggregation.

    `num_bases` below `num_rels` uses the basis decomposition from the R-GCN
    paper. `num_bases == 0` keeps a full matrix per relation, which trains
    more reliably on the few dozen relations left after pruning.
    """

    def __init__(self, in_dim, out_dim, num_rels, num_bases=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        use_basis = num_bases and num_bases < num_rels
        if use_basis:
            self.basis = nn.Parameter(torch.empty(num_bases, in_dim, out_dim))
            self.comb = nn.Parameter(torch.empty(num_rels, num_bases))
            self.weight = None
        else:
            self.basis = None
            self.comb = None
            self.weight = nn.Parameter(torch.empty(num_rels, in_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        # Identity messages: a node starts as itself plus the mean of its
        # neighbors, so a shared gene is visible to the decoder immediately.
        eye = None
        if self.in_dim == self.out_dim:
            eye = torch.eye(self.in_dim)
        if self.weight is not None:
            nn.init.zeros_(self.weight)
            if eye is not None:
                for r in range(self.num_rels):
                    self.weight.data[r].copy_(eye)
            else:
                nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.zeros_(self.basis)
            nn.init.zeros_(self.comb)
            if eye is not None:
                self.basis.data[0].copy_(eye)
                self.comb.data[:, 0] = 1
            else:
                nn.init.xavier_uniform_(self.basis)
                nn.init.xavier_uniform_(self.comb)
        nn.init.zeros_(self.root.weight)
        if eye is not None:
            self.root.weight.data.copy_(0.1 * eye)
        else:
            nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.bias)

    def _matrix(self, rel):
        if self.weight is not None:
            return self.weight[rel]
        return torch.einsum("b,bio->io", self.comb[rel], self.basis)

    def forward(self, x, edge_index, edge_type):
        src, dst = edge_index
        out = self.root(x) + self.bias
        if src.numel() == 0:
            return out
        n = x.size(0)
        flat = edge_type * n + dst
        counts = torch.zeros(self.num_rels * n, device=x.device, dtype=x.dtype)
        counts.index_add_(0, flat, torch.ones(src.numel(), device=x.device, dtype=x.dtype))
        norm = counts[flat].clamp(min=1).unsqueeze(1)
        acc = torch.zeros(n, self.out_dim, device=x.device, dtype=x.dtype)
        for rel in torch.unique(edge_type).tolist():
            take = edge_type == rel
            msg = (x[src[take]] @ self._matrix(rel)) / norm[take]
            acc = acc.index_add(0, dst[take], msg)
        return out + acc


class RGCN(nn.Module):
    def __init__(self, n_nodes, n_rels, n_types, dim=128, n_layers=2, dropout=0.1, n_bases=0):
        super().__init__()
        self.n_rels = n_rels
        self.n_layers = n_layers
        self.emb = nn.Embedding(n_nodes, dim)
        self.type_emb = nn.Embedding(n_types, dim)
        self.convs = nn.ModuleList(
            RelConv(dim, dim, 2 * n_rels, n_bases) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(n_layers))
        self.proj = nn.Linear(dim * (n_layers + 1), dim)
        self.final = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.rel = nn.Parameter(torch.empty(n_rels, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.1)
        nn.init.zeros_(self.type_emb.weight)
        nn.init.ones_(self.rel)
        # Start from the 1-hop representation. That is the drug/gene and
        # disease/gene aggregate, i.e. the shared-target signal.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        dim = self.rel.shape[1]
        eye = torch.eye(dim)
        with torch.no_grad():
            # 1-hop neighborhood is the drug/gene and disease/gene signal.
            # A little of the 2-hop state is mixed in so pathway hops can count.
            self.proj.weight[:, dim:2 * dim].copy_(eye)
            if self.n_layers >= 2:
                self.proj.weight[:, 2 * dim:3 * dim].copy_(0.25 * eye)

    def encode(self, edge_index, edge_type, node_type, edge_drop=0.0):
        if self.training and edge_drop > 0 and edge_index.numel() > 0:
            keep = torch.rand(edge_index.size(1), device=edge_index.device) >= edge_drop
            if int(keep.sum()) > 0:
                edge_index = edge_index[:, keep]
                edge_type = edge_type[keep]
        h = self.emb.weight + self.type_emb(node_type)
        parts = [h]
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            msg = norm(conv(h, edge_index, edge_type))
            parts.append(msg)
            h = h + msg
            if i + 1 < self.n_layers:
                h = self.drop(F.gelu(h))
        return self.final(self.proj(torch.cat(parts, dim=-1)))

    def score_candidates(self, hidden, heads, rel_id, candidates):
        """Cosine against every candidate. Shared across drugs, so AUPRC is meaningful.

        A free diagonal relation stretches some drugs' scores above other drugs'
        true hits and drops global AUPRC even when each drug is ranked correctly.
        `rel_id` is unused; auxiliary relations still use `score_pairs`.
        """
        del rel_id
        h = F.normalize(hidden[heads], dim=-1)
        c = F.normalize(hidden[candidates], dim=-1)
        return h @ c.transpose(0, 1)

    def score_pairs(self, hidden, heads, rels, tails):
        h = F.normalize(hidden[heads], dim=-1)
        rel = self.rel[rels]
        t = F.normalize(hidden[tails], dim=-1)
        if t.dim() == 2:
            return (h * rel * t).sum(-1)
        return (h.unsqueeze(1) * rel.unsqueeze(1) * t).sum(-1)


def build_message_graph(src, dst, rid, undirected, edge_index_pool, blocked_rids, n_rels):
    """Bidirectional message-passing graph over `edge_index_pool`.

    `blocked_rids` (the DRG-DIS relations) stay out, so the encoder cannot
    read the indication edge it is supposed to predict. Undirected edges reuse
    the same relation in both directions; directed edges get an inverse id.
    """
    s = src[edge_index_pool]
    d = dst[edge_index_pool]
    r = rid[edge_index_pool]
    u = undirected[edge_index_pool]
    if blocked_rids.numel():
        keep = ~torch.isin(r, blocked_rids)
        s, d, r, u = s[keep], d[keep], r[keep], u[keep]
    rev = torch.where(u, r, r + n_rels)
    triples = torch.stack([torch.cat([s, d]), torch.cat([d, s]), torch.cat([r, rev])], 1)
    triples = torch.unique(triples, dim=0)
    return triples[:, 0].contiguous(), triples[:, 1].contiguous(), triples[:, 2].contiguous()


def pair_mask(heads, pos_idx, known_heads, known_pos, n_cand):
    """True on candidates that are already known true tails, except the label itself."""
    mask = torch.zeros(heads.shape[0], n_cand, dtype=torch.bool, device=heads.device)
    if known_heads.numel() == 0:
        return mask
    match = heads[:, None] == known_heads[None, :]
    rows, cols = match.nonzero(as_tuple=True)
    mask[rows, known_pos[cols]] = True
    mask[torch.arange(heads.shape[0], device=heads.device), pos_idx] = False
    return mask


def indication_loss(scores, pos_idx, neg_mask, alpha=1.0, gamma=0.0, uniform=0.25):
    """Self-adversarial logistic loss, plus a uniform BCE term for AUPRC.

    `scores` is [batch, candidates], higher is better. `neg_mask` is true where
    a candidate must be ignored (another true indication of the same drug).
    """
    batch = torch.arange(scores.shape[0], device=scores.device)
    pos = scores[batch, pos_idx]
    finite = ~neg_mask
    finite[batch, pos_idx] = False
    weights_in = scores.masked_fill(~finite, -1e4)
    weights = torch.softmax(alpha * weights_in, dim=1) * finite.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    loss_pos = -F.logsigmoid(pos + gamma)
    neg_in = torch.where(finite, scores, torch.zeros_like(scores))
    loss_neg = -(weights * F.logsigmoid(-(neg_in + gamma))).sum(dim=1)
    loss = (loss_pos + loss_neg).mean()
    if uniform <= 0:
        return loss
    bce_pos = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
    bce_neg = F.binary_cross_entropy_with_logits(neg_in, torch.zeros_like(neg_in), reduction="none")
    bce_neg = (bce_neg * finite.float()).sum() / finite.float().sum().clamp(min=1)
    return loss + uniform * (bce_pos + bce_neg)


def average_precision(labels, scores):
    """Area under the precision-recall curve. One positive among N scores has AP = 1/rank."""
    labels = labels.float()
    total = labels.sum()
    if labels.numel() == 0 or total <= 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    hit = labels[order]
    tp = torch.cumsum(hit, 0)
    precision = tp / torch.arange(1, hit.numel() + 1, device=hit.device, dtype=torch.float)
    return float((precision * hit).sum() / total)


def ranks_from_scores(scores, pos_idx, filt):
    """Mid-ranks. Ties count as half a rank, filtered-out candidates are ignored."""
    batch = torch.arange(scores.shape[0], device=scores.device)
    true = scores[batch, pos_idx]
    seen = scores.masked_fill(filt, float("-inf"))
    seen[batch, pos_idx] = true
    greater = (seen > true.unsqueeze(1)).sum(dim=1).float()
    tied = ((seen == true.unsqueeze(1)).sum(dim=1).float() - 1).clamp(min=0)
    return 1.0 + greater + 0.5 * tied


def summarize_ranks(ranks):
    if ranks.numel() == 0:
        return {"n": 0, "AUPRC": float("nan"), "MRR": float("nan"), "H1": float("nan"),
                "H3": float("nan"), "H10": float("nan")}
    return {
        "n": int(ranks.numel()),
        "MRR": round(float((1.0 / ranks).mean()), 4),
        "H1": round(float((ranks <= 1).float().mean()), 4),
        "H3": round(float((ranks <= 3).float().mean()), 4),
        "H10": round(float((ranks <= 10).float().mean()), 4),
    }
