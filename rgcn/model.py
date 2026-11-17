"""Score held-out drug–disease edges from the rest of the graph.

A drug–disease indication is held out of the features. The score is how
specifically the drug and the disease share genes, a gene's neighbor, a
pathway, a biological process, or a phenotype. Common hubs count less than
a bridge that only a few diseases touch.

Averaging those neighbors into one embedding washes the shared gene out:
a disease with thousands of genes becomes a blur, and the true disease never
reaches the top of the list.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def unique_pairs(nodes, bridges):
    if nodes.numel() == 0:
        return nodes.long(), bridges.long()
    packed = torch.stack([nodes.long(), bridges.long()], 1)
    packed = torch.unique(packed, dim=0)
    return packed[:, 0].contiguous(), packed[:, 1].contiguous()


def cap_per_node(nodes, bridges, bridge_score, k):
    """Keep the k highest-scoring bridges for each node."""
    nodes, bridges = unique_pairs(nodes, bridges)
    if nodes.numel() == 0 or k <= 0:
        return nodes[:0], bridges[:0]
    score = bridge_score[bridges]
    order = torch.argsort(score, descending=True, stable=True)
    nodes, bridges = nodes[order], bridges[order]
    order = torch.argsort(nodes, stable=True)
    nodes, bridges = nodes[order], bridges[order]
    change = torch.ones(nodes.shape[0], dtype=torch.bool)
    if nodes.shape[0] > 1:
        change[1:] = nodes[1:] != nodes[:-1]
    idx = torch.arange(nodes.shape[0])
    starts = torch.cummax(torch.where(change, idx, torch.zeros_like(idx)), 0)[0]
    keep = (idx - starts) < k
    return nodes[keep], bridges[keep]


def orient(src, dst, node_type, type_a, type_b):
    """Return endpoints as (type_a, type_b), whichever way the edge was stored."""
    if src.numel() == 0:
        return src.long(), dst.long()
    forward = (node_type[src] == type_a) & (node_type[dst] == type_b)
    backward = (node_type[src] == type_b) & (node_type[dst] == type_a)
    take = forward | backward
    s, d = src[take], dst[take]
    left_a = node_type[s] == type_a
    return torch.where(left_a, s, d).long(), torch.where(left_a, d, s).long()


def both_ways(src, dst, node_type, type_a):
    if src.numel() == 0:
        return src.long(), dst.long()
    take = (node_type[src] == type_a) & (node_type[dst] == type_a) & (src != dst)
    s, d = src[take], dst[take]
    return torch.cat([s, d]).long(), torch.cat([d, s]).long()


def take_label(src, dst, rid, pool, label_ok):
    empty = torch.empty(0, dtype=torch.long)
    if pool.numel() == 0 or not bool(label_ok.any()):
        return empty, empty
    s, d, r = src[pool], dst[pool], rid[pool]
    keep = label_ok[r]
    return s[keep].long(), d[keep].long()


def label_masks(rel_names):
    labels = [name.split("|", 1)[0] for name in rel_names]
    wanted = (
        "DRG-GEN", "DIS-GEN", "GEN-GEN", "DRG-PHE", "DIS-PHE",
        "PHE-GEN", "PWY-GEN", "BPO-GEN",
    )
    return {name: torch.tensor([lab == name for lab in labels], dtype=torch.bool) for name in wanted}


def neighbor_table(src, dst, n_nodes, cap, bridge_score):
    src, dst = cap_per_node(src, dst, bridge_score, cap)
    table = torch.full((n_nodes, cap), -1, dtype=torch.long)
    if src.numel() == 0:
        return table
    order = torch.argsort(src, stable=True)
    src, dst = src[order], dst[order]
    change = torch.ones(src.shape[0], dtype=torch.bool)
    if src.shape[0] > 1:
        change[1:] = src[1:] != src[:-1]
    idx = torch.arange(src.shape[0])
    starts = torch.cummax(torch.where(change, idx, torch.zeros_like(idx)), 0)[0]
    rank = idx - starts
    ok = rank < cap
    table[src[ok], rank[ok]] = dst[ok]
    return table


def expand(nodes, keys, table):
    empty = torch.empty(0, dtype=torch.long)
    if nodes.numel() == 0:
        return empty, empty
    picked = table[keys]
    ok = picked >= 0
    if not bool(ok.any()):
        return empty, empty
    return nodes.unsqueeze(1).expand_as(picked)[ok].long(), picked[ok].long()


def drop_pairs(nodes, bridges, drop_nodes, drop_bridges):
    """Remove pairs that also appear in the second list."""
    if nodes.numel() == 0 or drop_nodes.numel() == 0:
        return nodes.long(), bridges.long()
    width = int(torch.maximum(bridges.max(), drop_bridges.max())) + 1
    drop_key = torch.unique(drop_nodes.long() * width + drop_bridges.long())
    key = nodes.long() * width + bridges.long()
    pos = torch.searchsorted(drop_key, key)
    pos_c = pos.clamp(max=drop_key.numel() - 1)
    hit = (pos < drop_key.numel()) & (drop_key[pos_c] == key)
    return nodes[~hit].long(), bridges[~hit].long()


class OverlapChannel(nn.Module):
    """Weighted count of shared bridges. One shared bridge adds its idf."""

    def __init__(self, name, drug_node, drug_bridge, dis_row, dis_bridge, idf, n_nodes, n_dis):
        super().__init__()
        self.name = name
        self.n_nodes = int(n_nodes)
        self.n_bridge = int(idf.numel())
        self.n_dis = int(n_dis)
        self.register_buffer("drug_node", drug_node.long(), persistent=False)
        self.register_buffer("drug_bridge", drug_bridge.long(), persistent=False)
        order = torch.argsort(dis_bridge.long())
        self.register_buffer("rows_sorted", dis_row.long()[order], persistent=False)
        bridges_sorted = dis_bridge.long()[order]
        counts = torch.bincount(bridges_sorted, minlength=self.n_bridge)
        starts = torch.cumsum(counts, 0) - counts
        self.register_buffer("counts", counts.long(), persistent=False)
        self.register_buffer("starts", starts.long(), persistent=False)
        self.register_buffer("idf", idf.float())
        self.register_buffer("unit", idf.mean().clamp(min=1e-3).float())
        # 0 => each bridge starts at its idf. Training can turn a noisy bridge down.
        self.logit = nn.Parameter(torch.zeros(self.n_bridge))

    def bridge_weight(self):
        return self.idf * torch.exp(self.logit.clamp(-3.0, 2.0)) / self.unit

    def score(self, heads):
        if heads.numel() == 0:
            return torch.zeros(0, self.n_dis, device=heads.device)
        if heads.numel() > 128:
            parts = [self._score_chunk(heads[i:i + 128]) for i in range(0, heads.numel(), 128)]
            return torch.cat(parts, 0)
        return self._score_chunk(heads)

    def _score_chunk(self, heads):
        device = heads.device
        uniq, inv = torch.unique(heads, return_inverse=True)
        blank = torch.zeros(uniq.numel(), self.n_dis, device=device)
        if self.drug_node.numel() == 0 or self.rows_sorted.numel() == 0:
            return blank[inv]
        lookup = torch.full((self.n_nodes,), -1, dtype=torch.long, device=device)
        lookup[uniq] = torch.arange(uniq.numel(), device=device)
        pos = lookup[self.drug_node]
        hit = pos >= 0
        if not bool(hit.any()):
            return blank[inv]
        u = pos[hit]
        g = self.drug_bridge[hit]
        deg = self.counts[g]
        keep = deg > 0
        if not bool(keep.any()):
            return blank[inv]
        u, g, deg = u[keep], g[keep], deg[keep]
        w = self.bridge_weight()
        # One scatter stays small enough to backprop through. Split if a chunk
        # of drug edges fans out to more than a million disease links.
        pieces = []
        start = 0
        limit = 1_000_000
        while start < u.numel():
            csum = torch.cumsum(deg[start:], 0)
            n_fit = int((csum <= limit).sum().item())
            n_fit = max(n_fit, 1)
            sl = slice(start, start + n_fit)
            pieces.append(self._scatter(uniq.numel(), u[sl], g[sl], deg[sl], w))
            start += n_fit
        full = pieces[0] if len(pieces) == 1 else torch.stack(pieces, 0).sum(0)
        return full[inv]

    def _scatter(self, n_uniq, u, g, deg, w):
        total = int(deg.sum().item())
        flat = torch.zeros(n_uniq * self.n_dis, device=u.device)
        if total == 0:
            return flat.view(n_uniq, self.n_dis)
        u_rep = u.repeat_interleave(deg)
        g_rep = g.repeat_interleave(deg)
        group = (torch.cumsum(deg, 0) - deg).repeat_interleave(deg)
        within = torch.arange(total, device=u.device) - group
        rows = self.rows_sorted[self.starts[g].repeat_interleave(deg) + within]
        idx = u_rep.long() * self.n_dis + rows.long()
        flat = flat.scatter_add(0, idx, w[g_rep])
        return flat.view(n_uniq, self.n_dis)


class PathRanker(nn.Module):
    """Sum of overlap channels. The gene channel starts as the whole score."""

    def __init__(self, channels, dis_ids):
        super().__init__()
        if not channels:
            raise SystemExit(
                "no overlap features: the training graph never connects a drug and a "
                "disease through a gene, pathway, or phenotype"
            )
        self.channels = nn.ModuleList(channels)
        # softplus(-6) ≈ 0.0025. Enough to break an exact tie, not enough to
        # outvote one shared gene. Training can raise a channel that helps.
        init = torch.full((len(channels),), -6.0)
        for i, channel in enumerate(channels):
            if channel.name == "gene":
                init[i] = 0.55  # softplus(0.55) ≈ 1
        self.mix = nn.Parameter(init)
        self.register_buffer("dis_ids", dis_ids.long())

    def score_candidates(self, heads, candidates):
        weights = F.softplus(self.mix)
        parts = torch.stack([channel.score(heads) for channel in self.channels], 0)
        total = torch.einsum("c,cbd->bd", weights, parts)
        if candidates.data_ptr() == self.dis_ids.data_ptr():
            return total
        if candidates.numel() == self.dis_ids.numel() and torch.equal(candidates, self.dis_ids):
            return total
        pos = torch.searchsorted(self.dis_ids, candidates)
        return total.index_select(1, pos)


def _type(type_names, name):
    try:
        return type_names.index(name)
    except ValueError:
        return None


def make_channel(name, nodes, bridges, dis_ids, n_nodes):
    nodes, bridges = unique_pairs(nodes, bridges)
    if nodes.numel() == 0 or dis_ids.numel() == 0:
        return None
    if nodes.numel() > 3_000_000:
        print(f"channel {name}: capping {nodes.numel():,} links to 64 per node", flush=True)
        width = int(bridges.max()) + 1
        nodes, bridges = cap_per_node(nodes, bridges, torch.ones(width), 64)
    uniq, local = torch.unique(bridges, return_inverse=True)
    row = torch.searchsorted(dis_ids, nodes)
    in_range = row < dis_ids.numel()
    row_c = row.clamp(max=dis_ids.numel() - 1)
    is_dis = in_range & (dis_ids[row_c] == nodes)
    if int(is_dis.sum()) == 0 or int((~is_dis).sum()) == 0:
        return None
    pairs = torch.unique(torch.stack([row_c[is_dis], local[is_dis]], 1), dim=0)
    df = torch.bincount(pairs[:, 1], minlength=uniq.numel()).float()
    idf = 1.0 / torch.log(2.0 + df)
    print(
        f"channel {name}: {int((~is_dis).sum()):,} drug-side links, "
        f"{int(is_dis.sum()):,} disease-side links, {uniq.numel():,} bridges",
        flush=True,
    )
    return OverlapChannel(
        name,
        nodes[~is_dis], local[~is_dis],
        pairs[:, 0], pairs[:, 1],
        idf, n_nodes, dis_ids.numel(),
    )


def _reach(name, drug_n, drug_g, dis_n, dis_g, raw_s, raw_d, node_type, gene_t, term_t, n_nodes, dis_ids, cap):
    genes, terms = orient(raw_s, raw_d, node_type, gene_t, term_t)
    if genes.numel() == 0 or drug_n.numel() == 0 or dis_n.numel() == 0:
        return None
    specificity = 1.0 / torch.bincount(terms, minlength=n_nodes).float().clamp(min=1)
    table = neighbor_table(genes, terms, n_nodes, cap, specificity)
    reached_drug, drug_term = expand(drug_n, drug_g, table)
    reached_dis, dis_term = expand(dis_n, dis_g, table)
    if reached_drug.numel() == 0 or reached_dis.numel() == 0:
        return None
    return make_channel(
        name,
        torch.cat([reached_drug, reached_dis]),
        torch.cat([drug_term, dis_term]),
        dis_ids, n_nodes,
    )


def build_overlap_channels(src, dst, rid, pool, node_type, type_names, rel_names, dis_ids, n_nodes):
    """Channels from training edges. `pool` must not contain held-out indications."""
    print("building overlap channels from the training graph", flush=True)
    drug_t, dis_t, gene_t = _type(type_names, "DRG"), _type(type_names, "DIS"), _type(type_names, "GEN")
    if drug_t is None or dis_t is None or gene_t is None:
        raise SystemExit(f"node types must include DRG, DIS, and GEN, found {type_names}")
    masks = label_masks(rel_names)
    drug_n, drug_g = orient(*take_label(src, dst, rid, pool, masks["DRG-GEN"]), node_type, drug_t, gene_t)
    dis_n, dis_g = orient(*take_label(src, dst, rid, pool, masks["DIS-GEN"]), node_type, dis_t, gene_t)
    channels = []
    gene = make_channel(
        "gene",
        torch.cat([drug_n, dis_n]),
        torch.cat([drug_g, dis_g]),
        dis_ids, n_nodes,
    )
    if gene is not None:
        channels.append(gene)

    gg_s, gg_d = take_label(src, dst, rid, pool, masks["GEN-GEN"])
    gg_a, gg_b = both_ways(gg_s, gg_d, node_type, gene_t)
    if gg_a.numel() and drug_n.numel() and dis_n.numel():
        specificity = 1.0 / torch.bincount(gg_a, minlength=n_nodes).float().clamp(min=1)
        table = neighbor_table(gg_a, gg_b, n_nodes, 16, specificity)
        hop_n, hop_g = expand(drug_n, drug_g, table)
        hop_n, hop_g = drop_pairs(hop_n, hop_g, drug_n, drug_g)
        if hop_n.numel():
            two = make_channel(
                "gene2",
                torch.cat([hop_n, dis_n]),
                torch.cat([hop_g, dis_g]),
                dis_ids, n_nodes,
            )
            if two is not None:
                channels.append(two)

    for name, label, term_name, cap in (
        ("pathway", "PWY-GEN", "PWY", 32),
        ("process", "BPO-GEN", "BPO", 32),
    ):
        term_t = _type(type_names, term_name)
        if term_t is None:
            continue
        raw_s, raw_d = take_label(src, dst, rid, pool, masks[label])
        reached = _reach(
            name, drug_n, drug_g, dis_n, dis_g, raw_s, raw_d,
            node_type, gene_t, term_t, n_nodes, dis_ids, cap,
        )
        if reached is not None:
            channels.append(reached)

    phe_t = _type(type_names, "PHE")
    if phe_t is not None:
        phe_nodes, phe_bridges = [], []
        for label, owner in (("DRG-PHE", drug_t), ("DIS-PHE", dis_t)):
            raw_s, raw_d = take_label(src, dst, rid, pool, masks[label])
            owner_n, phe = orient(raw_s, raw_d, node_type, owner, phe_t)
            if owner_n.numel():
                phe_nodes.append(owner_n)
                phe_bridges.append(phe)
        raw_s, raw_d = take_label(src, dst, rid, pool, masks["PHE-GEN"])
        genes, phenos = orient(raw_s, raw_d, node_type, gene_t, phe_t)
        if genes.numel() and drug_n.numel() and dis_n.numel():
            specificity = 1.0 / torch.bincount(phenos, minlength=n_nodes).float().clamp(min=1)
            table = neighbor_table(genes, phenos, n_nodes, 32, specificity)
            for owners, keys in ((drug_n, drug_g), (dis_n, dis_g)):
                reached_n, reached_b = expand(owners, keys, table)
                if reached_n.numel():
                    phe_nodes.append(reached_n)
                    phe_bridges.append(reached_b)
        if phe_nodes:
            pheno = make_channel(
                "pheno", torch.cat(phe_nodes), torch.cat(phe_bridges), dis_ids, n_nodes,
            )
            if pheno is not None:
                channels.append(pheno)

    if not any(channel.name == "gene" for channel in channels):
        print("warning: no direct drug–gene–disease channel. Check DRG-GEN and DIS-GEN edges.", flush=True)
    return channels


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
