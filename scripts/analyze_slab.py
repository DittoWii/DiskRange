#!/usr/bin/env python3
"""Analyze slab_neighbour_indices distribution to judge A2 (sparse-gather) feasibility.

Run: uv run --with numpy python scripts/analyze_slab.py
Requires /tmp/slab_neighbour.bin dumped by search with DJ_DUMP_SLAB=1.
"""
import numpy as np

with open("/tmp/slab_neighbour.bin", "rb") as f:
    cn, sc = (int(x) for x in np.frombuffer(f.read(16), dtype=np.uint64))
    nb = np.frombuffer(f.read(), dtype=np.uint32).reshape(cn, sc)

print(f"cluster_num={cn}  slab_count={sc}  array={nb.nbytes/1e6:.0f}MB")
rng = np.random.default_rng(42)

# 1. Per-cluster neighbour id spread (how scattered are the 512 neighbours in id space)
samp = rng.choice(cn, 2000, replace=False)
rng_span = (nb[samp].max(1) - nb[samp].min(1))
print(f"\n[1] neighbour id span (max-min) per cluster:")
print(f"    median={np.median(rng_span):.0f}  mean={rng_span.mean():.0f}"
      f"  => spans {100*np.median(rng_span)/cn:.0f}% of id space (neighbours fully scattered)")

# 2. distinct q_dot needed after dedup, for random vs spatially-clustered candidate sets
def distinct(cands):
    s = set(cands.tolist())
    for c in cands:
        s.update(nb[int(c)].tolist())
    return len(s)

randd = [distinct(rng.choice(cn, 245, replace=False)) for _ in range(20)]
print(f"\n[2a] random 245 candidates -> distinct q_dot:"
      f" median={int(np.median(randd))} range=[{min(randd)},{max(randd)}]"
      f" ({100*np.median(randd)/cn:.0f}% of {cn})")

# Spatially-clustered candidates ~ what SQG actually returns (seed + its nearest slab neighbours)
clud = []
for _ in range(20):
    seed = int(rng.integers(cn))
    cands = np.concatenate([[seed], nb[seed][:244].astype(np.int64)])
    clud.append(distinct(cands))
md = int(np.median(clud))
print(f"[2b] clustered 245 candidates -> distinct q_dot:"
      f" median={md} range=[{min(clud)},{max(clud)}]"
      f" ({100*md/cn:.0f}% of {cn});  measured actual mean=7644 (15%)")

# 3. gather footprint vs full scan (each centroid row = 96 float32 = 384B)
row_b = 96 * 4
print(f"\n[3] memory traffic (centroid row = {row_b}B):")
print(f"    full scan : {cn} rows = {cn*row_b/1e6:.1f}MB (current)")
print(f"    sparse A2 : {md} rows = {md*row_b/1e6:.2f}MB  ({100*md/cn:.0f}% traffic, {cn/md:.1f}x less)")
print(f"    but sparse = {md} RANDOM row gathers across an {cn*row_b/1e6:.0f}MB array")
