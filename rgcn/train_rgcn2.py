import torch, time, argparse, json
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
from torch_geometric.nn import RGCNConv

O = "/local/scratch/kkim636/"
p = argparse.ArgumentParser()
p.add_argument("--keep_nodes", default=None)
p.add_argument("--evalset", default=O + "evalset.pt")
p.add_argument("--tag", default="full")
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--bs", type=int, default=2048)
p.add_argument("--neg", type=int, default=2)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--bases", type=int, default=16)
p.add_argument("--fan", type=int, nargs=2, default=[10, 5])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval_every", type=int, default=2000)
p.add_argument("--mp_frac", type=float, default=0.5)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_test", action="store_true")
args = p.parse_args()

dev = torch.device("cuda")
torch.manual_seed(args.seed)

d = torch.load(O + "okg_split.pt")
N, R, T = d["num_nodes"], d["num_rels"], d["num_ntypes"]
src, dst, rid = d["src"], d["dst"], d["rid"]
node_type = d["node_type"].to(dev)

K = torch.ones(N, dtype=torch.bool)
if args.keep_nodes:
    idx = pl.read_parquet(O + "node_index.parquet").select("id", "nid")
    ids = pl.read_parquet(args.keep_nodes).select("id").unique()
    K = torch.zeros(N, dtype=torch.bool)
    K[torch.from_numpy(idx.join(ids, on="id")["nid"].to_numpy().astype("int64"))] = True

tr = d["split"]["train"]
tr = tr[K[src[tr]] & K[dst[tr]]]
tr = tr[torch.randperm(tr.numel())]
n_mp = int(args.mp_frac * tr.numel())
mp_idx, sup_idx = tr[:n_mp], tr[n_mp:]
print(f"[{args.tag}] keep nodes {int(K.sum())}/{N} | train edges {tr.numel()} | mp {mp_idx.numel()} sup {sup_idx.numel()}", flush=True)

ms, md, mr = src[mp_idx].to(dev), dst[mp_idx].to(dev), rid[mp_idx].to(dev)
a_src = torch.cat([ms, md]); a_dst = torch.cat([md, ms]); a_rel = torch.cat([mr, mr + R])
order_e = torch.argsort(a_dst)
col, et = a_src[order_e], a_rel[order_e]
deg = torch.bincount(a_dst, minlength=N)
rowptr = torch.zeros(N + 1, dtype=torch.long, device=dev)
rowptr[1:] = torch.cumsum(deg, 0)
del a_src, a_dst, a_rel, order_e, ms, md, mr
sup_s, sup_d, sup_r = src[sup_idx].to(dev), dst[sup_idx].to(dev), rid[sup_idx].to(dev)

kn = K.to(dev).nonzero().squeeze(1)
kt = node_type[kn]
k_sorted = kn[torch.argsort(kt)]
cnt = torch.bincount(kt, minlength=T)
start = torch.cumsum(cnt, 0) - cnt

def sample_neg(true_dst, k):
    t = node_type[true_dst].unsqueeze(1).expand(-1, k)
    r = (torch.rand(true_dst.numel(), k, device=dev) * cnt[t]).long()
    r = torch.minimum(r, cnt[t] - 1)
    return k_sorted[start[t] + r]

def sample_nbrs(nodes, f):
    st = rowptr[nodes]
    dg = rowptr[nodes + 1] - st
    r = (torch.rand(nodes.numel(), f, device=dev) * dg.unsqueeze(1)).long()
    r = torch.minimum(r, (dg - 1).clamp(min=0).unsqueeze(1))
    ix = st.unsqueeze(1) + r
    m = (dg > 0).unsqueeze(1).expand(-1, f)
    return col[ix[m]], nodes.unsqueeze(1).expand(-1, f)[m], et[ix[m]]

def subgraph(seeds):
    seeds = torch.unique(seeds)
    es, ed, er = [], [], []
    frontier = seeds
    for f in args.fan:
        s, t, r = sample_nbrs(frontier, f)
        es.append(s); ed.append(t); er.append(r)
        frontier = torch.unique(s)
    all_s, all_t, all_r = torch.cat(es), torch.cat(ed), torch.cat(er)
    nodes = torch.unique(torch.cat([seeds, all_s]))
    lut = torch.full((N,), -1, dtype=torch.long, device=dev)
    lut[nodes] = torch.arange(nodes.numel(), device=dev)
    return nodes, torch.stack([lut[all_s], lut[all_t]]), all_r, lut

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(N, args.dim)
        s.c1 = RGCNConv(args.dim, args.dim, 2 * R, num_bases=args.bases)
        s.c2 = RGCNConv(args.dim, args.dim, 2 * R, num_bases=args.bases)
        s.rel = nn.Parameter(torch.empty(R, args.dim))
        nn.init.xavier_uniform_(s.rel)
    def encode(s, nodes, ei, er):
        x = s.emb(nodes)
        x = F.relu(s.c1(x, ei, er))
        return s.c2(x, ei, er)

model = Net().to(dev)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

rels = pl.read_parquet(O + "rel_index.parquet")
ind_rid = rels.filter(pl.col("rtype") == "DRG-DIS|INDICATION")["rid"][0]
EV = torch.load(args.evalset)

def evaluate(name, chunk=128):
    ev = EV[name]
    s, t, r, neg, valid = [ev[x].to(dev) for x in ("s", "t", "r", "neg", "valid")]
    tn = ev["tail_neuro"].to(dev)
    model.eval()
    ranks = []
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(1234)
        for i in range(0, s.numel(), chunk):
            ss, tt, rr_, ng, vv = s[i:i+chunk], t[i:i+chunk], r[i:i+chunk], neg[i:i+chunk], valid[i:i+chunk]
            nodes, ei, er, lut = subgraph(torch.cat([ss, tt, ng.flatten()]))
            h = model.encode(nodes, ei, er)
            hs, ht, hn = h[lut[ss]], h[lut[tt]], h[lut[ng]]
            rv = model.rel[rr_]
            pos = (hs * rv * ht).sum(-1)
            sc = (hs.unsqueeze(1) * rv.unsqueeze(1) * hn).sum(-1).masked_fill(~vv, float("-inf"))
            ranks.append(1 + (sc > pos.unsqueeze(1)).sum(1).float())
    ranks = torch.cat(ranks)
    model.train()
    ind = r == ind_rid
    out = {}
    for nm, m in [("all", torch.ones_like(ind)), ("INDICATION", ind), ("INDICATION_neuro", ind & tn)]:
        if int(m.sum()) > 0:
            rk = ranks[m]
            out[nm] = {"n": int(m.sum()), "MRR": round((1 / rk).mean().item(), 4),
                       "H1": round((rk <= 1).float().mean().item(), 4),
                       "H10": round((rk <= 10).float().mean().item(), 4)}
    return out

t0 = time.time()
res = {"args": vars(args), "keep_nodes": int(K.sum()), "train_edges": int(tr.numel()), "val_curve": {}}
for step in range(1, args.steps + 1):
    b = torch.randint(0, sup_s.numel(), (args.bs,), device=dev)
    s, t, r = sup_s[b], sup_d[b], sup_r[b]
    ng = sample_neg(t, args.neg)
    nodes, ei, er, lut = subgraph(torch.cat([s, t, ng.flatten()]))
    h = model.encode(nodes, ei, er)
    hs, ht, hn = h[lut[s]], h[lut[t]], h[lut[ng]]
    rv = model.rel[r]
    pos = (hs * rv * ht).sum(-1)
    neg = (hs.unsqueeze(1) * rv.unsqueeze(1) * hn).sum(-1)
    loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f} {(time.time()-t0)/step:.3f}s/step", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        v = evaluate("val")
        res["val_curve"][step] = v
        print(f"[val] step {step} {json.dumps(v)}", flush=True)

res["val"] = res["val_curve"][args.steps]
if args.eval_test:
    res["test"] = evaluate("test")
    print("[test]", json.dumps(res["test"]), flush=True)
torch.save(model.state_dict(), O + f"rgcn_{args.tag}.pt")
json.dump(res, open(O + f"res_{args.tag}.json", "w"), indent=1)
print("saved", args.tag)
