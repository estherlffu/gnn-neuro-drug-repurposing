
"""Memory-efficient R-GCN encoder and drug-disease scoring.

Baseline-specific version of model.py.

Uses chunked relational message aggregation to reduce peak GPU memory
on the full, unpruned OptimusKG graph. The full-graph per-relation
destination degrees are retained, so aggregation normalization is
unchanged.

All model architecture, scoring, loss, and graph construction logic
matches model.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Memory-efficient chunked aggregation
# ------------------------------------------------------------

class ChunkedRelationalAggregate(torch.autograd.Function):
    """Memory-bounded relational aggregation with full-graph degrees.

    Messages are computed in chunks during the forward pass.
    During backward, messages are recomputed rather than retained.

    This preserves the same per-relation mean aggregation as the
    original RelConv.forward().
    """

    @staticmethod
    def forward(
        ctx,
        x,
        src,
        dst,
        edge_type,
        weights,
        num_rels,
        chunk_size,
    ):
        n = x.size(0)
        n_edges = src.numel()

        # Compute full-graph destination counts separately for each
        # relation. These counts are shared across all edge chunks.
        flat = edge_type * n + dst

        counts = torch.zeros(
            num_rels * n,
            device=x.device,
            dtype=x.dtype,
        )

        counts.index_add_(
            0,
            flat,
            torch.ones(
                n_edges,
                device=x.device,
                dtype=x.dtype,
            ),
        )

        # Aggregate messages in chunks.
        out = torch.zeros(
            n,
            weights.size(-1),
            device=x.device,
            dtype=x.dtype,
        )

        for start in range(0, n_edges, chunk_size):
            end = min(start + chunk_size, n_edges)

            s_chunk = src[start:end]
            d_chunk = dst[start:end]
            r_chunk = edge_type[start:end]

            for rel in torch.unique(r_chunk).tolist():
                mask = r_chunk == rel

                s = s_chunk[mask]
                d = d_chunk[mask]

                norm = counts[rel * n + d].clamp(min=1).unsqueeze(1)

                msg = (x[s] @ weights[rel]) / norm

                out.index_add_(0, d, msg)

        ctx.num_rels = num_rels
        ctx.chunk_size = chunk_size

        ctx.save_for_backward(
            x,
            src,
            dst,
            edge_type,
            weights,
            counts,
        )

        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, src, dst, edge_type, weights, counts = ctx.saved_tensors

        n = x.size(0)
        n_edges = src.numel()

        grad_x = torch.zeros_like(x)
        grad_weights = torch.zeros_like(weights)

        # Recompute messages chunk by chunk during backward.
        for start in range(0, n_edges, ctx.chunk_size):
            end = min(start + ctx.chunk_size, n_edges)

            s_chunk = src[start:end]
            d_chunk = dst[start:end]
            r_chunk = edge_type[start:end]

            for rel in torch.unique(r_chunk).tolist():
                mask = r_chunk == rel

                s = s_chunk[mask]
                d = d_chunk[mask]

                norm = counts[rel * n + d].clamp(min=1).unsqueeze(1)

                g = grad_out[d] / norm

                # Gradient with respect to source node embeddings.
                grad_x.index_add_(
                    0,
                    s,
                    g @ weights[rel].T,
                )

                # Gradient with respect to the relation matrix.
                grad_weights[rel] += x[s].T @ g

        return (
            grad_x,
            None,
            None,
            None,
            grad_weights,
            None,
            None,
        )


# ------------------------------------------------------------
# Relational graph convolution
# ------------------------------------------------------------

class RelConv(nn.Module):
    """Relational graph convolution with per-relation mean aggregation."""

    def __init__(self, in_dim, out_dim, num_rels, num_bases=0):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels

        self.root = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        use_basis = num_bases and num_bases < num_rels

        if use_basis:
            self.basis = nn.Parameter(
                torch.empty(num_bases, in_dim, out_dim)
            )
            self.comb = nn.Parameter(
                torch.empty(num_rels, num_bases)
            )
            self.weight = None
        else:
            self.basis = None
            self.comb = None
            self.weight = nn.Parameter(
                torch.empty(num_rels, in_dim, out_dim)
            )

        self.reset_parameters()

    def reset_parameters(self):
        # Identity messages: a node starts as itself plus the mean
        # of its neighbors.
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

        return torch.einsum(
            "b,bio->io",
            self.comb[rel],
            self.basis,
        )

    def forward(self, x, edge_index, edge_type):
        src, dst = edge_index

        out = self.root(x) + self.bias

        if src.numel() == 0:
            return out

        # Obtain the effective relation matrices.
        if self.weight is not None:
            weights = self.weight
        else:
            weights = torch.stack([
                self._matrix(r)
                for r in range(self.num_rels)
            ])

        # Process edges in bounded chunks.
        # Full-graph relation/destination counts are preserved.
        acc = ChunkedRelationalAggregate.apply(
            x,
            src,
            dst,
            edge_type,
            weights,
            self.num_rels,
            100_000,
        )

        return out + acc


# ------------------------------------------------------------
# R-GCN model
# ------------------------------------------------------------

class RGCN(nn.Module):

    def __init__(
        self,
        n_nodes,
        n_rels,
        n_types,
        dim=128,
        n_layers=2,
        dropout=0.1,
        n_bases=0,
    ):
        super().__init__()

        self.n_rels = n_rels
        self.n_layers = n_layers

        self.emb = nn.Embedding(n_nodes, dim)
        self.type_emb = nn.Embedding(n_types, dim)

        self.convs = nn.ModuleList(
            RelConv(dim, dim, 2 * n_rels, n_bases)
            for _ in range(n_layers)
        )

        self.norms = nn.ModuleList(
            nn.LayerNorm(dim)
            for _ in range(n_layers)
        )

        self.proj = nn.Linear(dim * (n_layers + 1), dim)
        self.final = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

        self.rel = nn.Parameter(
            torch.empty(n_rels, dim)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.1)
        nn.init.zeros_(self.type_emb.weight)
        nn.init.ones_(self.rel)

        # Start from the 1-hop representation.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        dim = self.rel.shape[1]
        eye = torch.eye(dim)

        with torch.no_grad():
            self.proj.weight[:, dim:2 * dim].copy_(eye)

            if self.n_layers >= 2:
                self.proj.weight[:, 2 * dim:3 * dim].copy_(
                    0.25 * eye
                )

    def encode(
        self,
        edge_index,
        edge_type,
        node_type,
        edge_drop=0.0,
    ):
        if (
            self.training
            and edge_drop > 0
            and edge_index.numel() > 0
        ):
            keep = (
                torch.rand(
                    edge_index.size(1),
                    device=edge_index.device,
                ) >= edge_drop
            )

            if int(keep.sum()) > 0:
                edge_index = edge_index[:, keep]
                edge_type = edge_type[keep]

        h = self.emb.weight + self.type_emb(node_type)

        parts = [h]

        for i, (conv, norm) in enumerate(
            zip(self.convs, self.norms)
        ):
            msg = norm(
                conv(h, edge_index, edge_type)
            )

            parts.append(msg)

            h = h + msg

            if i + 1 < self.n_layers:
                h = self.drop(F.gelu(h))

        return self.final(
            self.proj(torch.cat(parts, dim=-1))
        )

    def score_candidates(
        self,
        hidden,
        heads,
        rel_id,
        candidates,
    ):
        """Cosine similarity against every candidate disease."""

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

        return (
            h.unsqueeze(1)
            * rel.unsqueeze(1)
            * t
        ).sum(-1)


# ------------------------------------------------------------
# Message graph construction
# ------------------------------------------------------------

def build_message_graph(
    src,
    dst,
    rid,
    undirected,
    edge_index_pool,
    blocked_rids,
    n_rels,
):
    """Bidirectional message-passing graph over edge_index_pool.

    DRG-DIS relations are blocked to prevent information leakage.
    Undirected edges reuse the same relation in both directions.
    Directed edges receive an inverse relation ID.
    """

    s = src[edge_index_pool]
    d = dst[edge_index_pool]
    r = rid[edge_index_pool]
    u = undirected[edge_index_pool]

    if blocked_rids.numel():
        keep = ~torch.isin(r, blocked_rids)
        s, d, r, u = s[keep], d[keep], r[keep], u[keep]

    rev = torch.where(u, r, r + n_rels)

    triples = torch.stack(
        [
            torch.cat([s, d]),
            torch.cat([d, s]),
            torch.cat([r, rev]),
        ],
        1,
    )

    triples = torch.unique(triples, dim=0)

    return (
        triples[:, 0].contiguous(),
        triples[:, 1].contiguous(),
        triples[:, 2].contiguous(),
    )


# ------------------------------------------------------------
# Evaluation and loss utilities
# ------------------------------------------------------------

def pair_mask(
    heads,
    pos_idx,
    known_heads,
    known_pos,
    n_cand,
):
    """True on already-known positive tails, except the target label."""

    mask = torch.zeros(
        heads.shape[0],
        n_cand,
        dtype=torch.bool,
        device=heads.device,
    )

    if known_heads.numel() == 0:
        return mask

    match = heads[:, None] == known_heads[None, :]

    rows, cols = match.nonzero(as_tuple=True)

    mask[rows, known_pos[cols]] = True

    mask[
        torch.arange(heads.shape[0], device=heads.device),
        pos_idx,
    ] = False

    return mask


def indication_loss(
    scores,
    pos_idx,
    neg_mask,
    alpha=1.0,
    gamma=0.0,
    uniform=0.25,
):
    """Self-adversarial logistic loss plus uniform BCE."""

    batch = torch.arange(
        scores.shape[0],
        device=scores.device,
    )

    pos = scores[batch, pos_idx]

    finite = ~neg_mask
    finite[batch, pos_idx] = False

    weights_in = scores.masked_fill(~finite, -1e4)

    weights = (
        torch.softmax(alpha * weights_in, dim=1)
        * finite.float()
    )

    weights = weights / weights.sum(
        dim=1,
        keepdim=True,
    ).clamp(min=1e-12)

    loss_pos = -F.logsigmoid(pos + gamma)

    neg_in = torch.where(
        finite,
        scores,
        torch.zeros_like(scores),
    )

    loss_neg = -(
        weights
        * F.logsigmoid(-(neg_in + gamma))
    ).sum(dim=1)

    loss = (loss_pos + loss_neg).mean()

    if uniform <= 0:
        return loss

    bce_pos = F.binary_cross_entropy_with_logits(
        pos,
        torch.ones_like(pos),
    )

    bce_neg = F.binary_cross_entropy_with_logits(
        neg_in,
        torch.zeros_like(neg_in),
        reduction="none",
    )

    bce_neg = (
        (bce_neg * finite.float()).sum()
        / finite.float().sum().clamp(min=1)
    )

    return loss + uniform * (bce_pos + bce_neg)


def average_precision(labels, scores):
    """Area under the precision-recall curve."""

    labels = labels.float()
    total = labels.sum()

    if labels.numel() == 0 or total <= 0:
        return float("nan")

    order = torch.argsort(
        scores,
        descending=True,
        stable=True,
    )

    hit = labels[order]
    tp = torch.cumsum(hit, 0)

    precision = tp / torch.arange(
        1,
        hit.numel() + 1,
        device=hit.device,
        dtype=torch.float,
    )

    return float((precision * hit).sum() / total)


def ranks_from_scores(scores, pos_idx, filt):
    """Mid-ranks. Filtered candidates are ignored."""

    batch = torch.arange(
        scores.shape[0],
        device=scores.device,
    )

    true = scores[batch, pos_idx]

    seen = scores.masked_fill(
        filt,
        float("-inf"),
    )

    seen[batch, pos_idx] = true

    greater = (
        seen > true.unsqueeze(1)
    ).sum(dim=1).float()

    tied = (
        (seen == true.unsqueeze(1)).sum(dim=1).float() - 1
    ).clamp(min=0)

    return 1.0 + greater + 0.5 * tied


def summarize_ranks(ranks):
    if ranks.numel() == 0:
        return {
            "n": 0,
            "AUPRC": float("nan"),
            "MRR": float("nan"),
            "H1": float("nan"),
            "H3": float("nan"),
            "H10": float("nan"),
        }

    return {
        "n": int(ranks.numel()),
        "MRR": round(float((1.0 / ranks).mean()), 4),
        "H1": round(float((ranks <= 1).float().mean()), 4),
        "H3": round(float((ranks <= 3).float().mean()), 4),
        "H10": round(float((ranks <= 10).float().mean()), 4),
    }
