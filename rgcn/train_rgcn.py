import torch, time, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--bs", type=int, default=2048)
p.add_argument("--neg", type=int, default=2)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--bases", type=int, default=16)
p.add_argument("--fan", type=int, nargs=2, default=[10, 5])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval_every", type=int, default=2000)
p.add_argument("--mp_frac", type=float, default=0.5)
args = p.parse_args()

dev = torch.device("cuda")
torch.manual_seed(0)

d = torch.load("/local/scratch/kkim636/okg_split.pt")
N, R = d["num_nodes"], d["num_rels"]
src, dst, rid = d["src"], d["dst"], d["rid"]
node_type = d["node_type"].to(dev)

tr = d["split"]["train"]
tr = tr[torch.randperm(tr.numel())]
n_mp = int(args.mp_frac * tr.numel())
mp_idx, sup_idx = tr[:n_mp], tr[n_mp:]
val_idx = d["split"]["val"]

# message-passing graph (train half) + inverse relations, CSR by target node
ms, md, mr = src[mp_idx].to(dev), dst[mp_idx].to(dev), rid[mp_idx].to(dev)
a_src = torch.cat([ms, md]); a_dst = torch.cat([md, ms]); a_rel = torch.cat([mr, mr + R])
order_e = torch.argsort(a_dst)
col, et = a_src[order_e], a_rel[order_e]
deg = torch.bincount(a_dst, minlength=N)
rowptr = torch.zeros(N + 1, dtype=torch.long, device=dev)
rowptr[1:] = torch.cumsum(deg, 0)
del a_src, a_dst, a_rel, order_e, ms, md, mr

sup_s, sup_d, sup_r = src[sup_idx].to(dev), dst[sup_idx].to(dev), rid[sup_idx].to(dev)

# type-matched negative sampling
t_order = torch.argsort(node_type)
t_cnt = torch.bincount(node_type)
t_start = torch.cumsum(t_cnt, 0) - t_cnt

def sample_neg(true_dst, k):
    t = node_type[true_dst].unsqueeze(1).expand(-1, k)
    r = (torch.rand(true_dst.numel(), k, device=dev) * t_cnt[t]).long()
    r = torch.minimum(r, t_cnt[t] - 1)
    return t_order[t_start[t] + r]

def sample_nbrs(nodes, f):
    start = rowptr[nodes]
    dg = rowptr[nodes + 1] - start
    r = (torch.rand(nodes.numel(), f, device=dev) * dg.unsqueeze(1)).long()
    r = torch.minimum(r, (dg - 1).clamp(min=0).unsqueeze(1))
    idx = start.unsqueeze(1) + r
    m = (dg > 0).unsqueeze(1).expand(-1, f)
    return col[idx[m]], nodes.unsqueeze(1).expand(-1, f)[m], et[idx[m]]

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
    ei = torch.stack([lut[all_s], lut[all_t]])
    return nodes, ei, all_r, lut

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

def run_batch(s, t, r, k):
    ng = sample_neg(t, k)
    nodes, ei, er, lut = subgraph(torch.cat([s, t, ng.flatten()]))
    h = model.encode(nodes, ei, er)
    hs, ht, hn = h[lut[s]], h[lut[t]], h[lut[ng]]
    rr = model.rel[r]
    pos = (hs * rr * ht).sum(-1)
    neg = (hs.unsqueeze(1) * rr.unsqueeze(1) * hn).sum(-1)
    return pos, neg

@torch.no_grad()
def evaluate(n=5000, k=50, chunk=128):
    model.eval()
    idx = val_idx[:n]
    vs, vd, vr = src[idx].to(dev), dst[idx].to(dev), rid[idx].to(dev)
    ranks = []
    for i in range(0, vs.numel(), chunk):
        pos, neg = run_batch(vs[i:i+chunk], vd[i:i+chunk], vr[i:i+chunk], k)
        ranks.append(1 + (neg > pos.unsqueeze(1)).sum(1).float())
    ranks = torch.cat(ranks)
    model.train()
    return (1.0 / ranks).mean().item(), (ranks <= 10).float().mean().item()

print("nodes", N, "rels", R, "mp edges", mp_idx.numel(), "sup edges", sup_idx.numel(), flush=True)
t0 = time.time()
for step in range(1, args.steps + 1):
    b = torch.randint(0, sup_s.numel(), (args.bs,), device=dev)
    pos, neg = run_batch(sup_s[b], sup_d[b], sup_r[b], args.neg)
    loss = (F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f} {(time.time()-t0)/step:.3f}s/step", flush=True)
    if step % args.eval_every == 0 or step == args.steps:
        mrr, h10 = evaluate()
        print(f"[val] step {step} MRR {mrr:.4f} Hits@10 {h10:.4f}", flush=True)

torch.save(model.state_dict(), "/local/scratch/kkim636/rgcn.pt")
print("saved rgcn.pt")
