"""RotatE on the OptimusKG split, with the same interface and evaluation as train_rgcn2.py.

- Trains on train edges only (optionally restricted to a pruned node set via --keep_nodes).
- Type-matched negatives (tail corruption), sampled from the kept nodes, same as the R-GCN runs.
- Self-adversarial negative sampling loss from the RotatE paper.
- Evaluated on the shared eval set (evalset_*.pt) with the same metrics/groups as train_rgcn2.py.

The score function `rotate` is the one from the teammate's embedding.py (complex pairs, relation
normalized to modulus 1, L1 over |h*r - t|); a tiny epsilon is added under the sqrt so the gradient
is finite when a distance is exactly zero.
"""
import torch, time, argparse, json
import torch.nn as nn
import torch.nn.functional as F
import polars as pl

p = argparse.ArgumentParser()
p.add_argument("--root", default="/local/scratch/kkim636/")
p.add_argument("--keep_nodes", default=None)
p.add_argument("--evalset", default=None, help="default: <root>evalset.pt")
p.add_argument("--tag", default="full")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=2048)
p.add_argument("--neg", type=int, default=32, help="negatives per positive")
p.add_argument("--dim", type=int, default=128, help="real dimension (must be even): dim/2 complex numbers")
p.add_argument("--gamma", type=float, default=9.0, help="margin")
p.add_argument("--alpha", type=float, default=1.0, help="self-adversarial temperature")
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval_every", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--eval_test", action="store_true")
args = p.parse_args()
assert args.dim % 2 == 0, "RotatE needs an even dimension"

O = args.root
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
print(f"[{args.tag}] keep nodes {int(K.sum())}/{N} | train edges {tr.numel()}", flush=True)
sup_s, sup_d, sup_r = src[tr].to(dev), dst[tr].to(dev), rid[tr].to(dev)

# type-matched negative sampling restricted to kept nodes
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

def rotate(h, r, t):
    """RotatE distance. Last dim = [real parts | imaginary parts]. Smaller = better."""
    hr, hi = h.chunk(2, dim=-1)
    rr, ri = r.chunk(2, dim=-1)
    tr_, ti = t.chunk(2, dim=-1)
    mod = torch.hypot(rr, ri).clamp_min(1e-9)
    rr, ri = rr / mod, ri / mod
    dr = hr * rr - hi * ri - tr_
    di = hr * ri + hi * rr - ti
    return (dr * dr + di * di + 1e-12).sqrt().sum(dim=-1)

rng = (args.gamma + 2.0) / (args.dim // 2)
E = nn.Parameter(torch.empty(N, args.dim, device=dev).uniform_(-rng, rng))
Rel = nn.Parameter(torch.randn(R, args.dim, device=dev))
opt = torch.optim.Adam([E, Rel], lr=args.lr)

rels = pl.read_parquet(O + "rel_index.parquet")
ind_rid = rels.filter(pl.col("rtype") == "DRG-DIS|INDICATION")["rid"][0]
EV = torch.load(args.evalset or (O + "evalset.pt"))

def evaluate(name, chunk=512):
    ev = EV[name]
    s, t, r, neg, valid = [ev[x].to(dev) for x in ("s", "t", "r", "neg", "valid")]
    tn = ev["tail_neuro"].to(dev)
    ranks = []
    with torch.no_grad():
        for i in range(0, s.numel(), chunk):
            ss, tt, rr_, ng, vv = s[i:i+chunk], t[i:i+chunk], r[i:i+chunk], neg[i:i+chunk], valid[i:i+chunk]
            hs, rv = E[ss], Rel[rr_]
            pos = -rotate(hs, rv, E[tt])
            sc = (-rotate(hs.unsqueeze(1), rv.unsqueeze(1), E[ng])).masked_fill(~vv, float("-inf"))
            ranks.append(1 + (sc > pos.unsqueeze(1)).sum(1).float())
    ranks = torch.cat(ranks)
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
    hs, rv = E[s], Rel[r]
    pos_d = rotate(hs, rv, E[t])                                          # [B]
    neg_d = rotate(hs.unsqueeze(1), rv.unsqueeze(1), E[ng])               # [B, k]
    w = torch.softmax(-args.alpha * neg_d, dim=1).detach()                # self-adversarial weights
    loss = (-F.logsigmoid(args.gamma - pos_d)
            - (w * F.logsigmoid(neg_d - args.gamma)).sum(dim=1)).mean()
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
torch.save({"E": E.data.cpu(), "R": Rel.data.cpu(), "dim": args.dim, "gamma": args.gamma},
           O + f"rotate_{args.tag}.pt")
json.dump(res, open(O + f"res_rotate_{args.tag}.json", "w"), indent=1)
print("saved", args.tag)