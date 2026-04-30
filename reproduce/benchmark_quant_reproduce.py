#!/usr/bin/env python3
"""Reproduce TurboQuant Fig. 5 (paper §4.4) — GloVe-200 only, TurboQuant only.

Paper setup (Section 4.4):
  - Base:  100,000 randomly sampled from GloVe-200 train (1,183,514)
  - Query: pre-existing 10,000-vector test set
  - Metric: Recall@1@k = P(true top-1-by-IP ∈ approx top-k-by-IP)
  - k ∈ {1, 2, 4, 8, 16, 32, 64}
  - Bit-widths: 2 bit/dim and 4 bit/dim

Reuses TurboQuantMSE / TurboQuantProd / load_dataset from benchmark_quant.py.
"""

import argparse
import csv
import os
import time

import numpy as np

from benchmark_quant import TurboQuantMSE, TurboQuantProd, load_dataset

PAPER_KS = [1, 2, 4, 8, 16, 32, 64]


def recall_at_1_at_k(approx_ip, exact_ip, ks):
    """Recall@1@k under inner product.

    approx_ip, exact_ip: [nb, nq] matrices of base-vs-query IPs.
    Returns {k: P(argmax_i exact_ip[:, q]  ∈  top-k of approx_ip[:, q])}.
    """
    nb, nq = exact_ip.shape
    true_top1 = np.argmax(exact_ip, axis=0)              # [nq]
    rank = np.argsort(-approx_ip, axis=0)                # [nb, nq], desc by approx IP
    out = {}
    for k in ks:
        if k > nb:
            out[k] = float("nan")
            continue
        topk = rank[:k, :]                                # [k, nq]
        out[k] = float(np.mean(np.any(topk == true_top1[None, :], axis=0)))
    return out


def run_method(method, base, queries, exact_ip, ks):
    t0 = time.time()
    code = method.encode(base)
    enc_t = time.time() - t0

    t0 = time.time()
    approx_ip = method.estimate_ip(code, queries)
    est_t = time.time() - t0

    rec = recall_at_1_at_k(approx_ip, exact_ip, ks)
    return {
        "method": method.name,
        "encode_s": enc_t,
        "estimate_s": est_t,
        **{f"R@1@{k}": rec[k] for k in ks},
    }


def main():
    p = argparse.ArgumentParser(description="Reproduce TurboQuant Fig. 5 on GloVe-200.")
    p.add_argument("--dataset", default="glove")
    p.add_argument("--nb", type=int, default=100_000, help="base size (paper: 100,000)")
    p.add_argument("--nq", type=int, default=10_000, help="query size (paper GloVe: 10,000)")
    p.add_argument("--bits", type=int, nargs="+", default=[2, 4])
    p.add_argument("--ks", type=int, nargs="+", default=PAPER_KS)
    p.add_argument("--variant", choices=["mse", "prod", "both"], default="both",
                   help="TurboQuant variant: TQ-MSE, TQ-Prod (IP-unbiased), or both.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    base, queries, d = load_dataset(args.dataset, nb=args.nb, nq=args.nq)

    print(f"\nComputing exact IP ({base.shape[0]} × {queries.shape[0]} pairs)...")
    t0 = time.time()
    exact_ip = base @ queries.T
    print(f"  {time.time() - t0:.2f}s")

    methods = []
    for b in args.bits:
        if args.variant in ("mse", "both"):
            methods.append(TurboQuantMSE(d, b, seed=args.seed))
        if args.variant in ("prod", "both") and b >= 2:
            methods.append(TurboQuantProd(d, b, seed=args.seed))

    print(f"\nEvaluating {len(methods)} method(s) on '{args.dataset}' "
          f"d={d}, nb={base.shape[0]}, nq={queries.shape[0]}")
    rows = []
    for m in methods:
        print(f"  {m.name}...", end="", flush=True)
        r = run_method(m, base, queries, exact_ip, args.ks)
        rows.append(r)
        print(f" enc={r['encode_s']:.1f}s  est={r['estimate_s']:.1f}s")

    # Pretty print
    cw_k = max(7, max(len(f"R@1@{k}") for k in args.ks) + 1)
    hdr = f"  {'method':<16}" + "".join(f" {('R@1@'+str(k)):>{cw_k}}" for k in args.ks)
    print("\n" + "=" * len(hdr))
    print(f"  Recall@1@k on {args.dataset} (d={d}, nb={base.shape[0]}, nq={queries.shape[0]})")
    print("=" * len(hdr))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(f"  {r['method']:<16}" + "".join(f" {r[f'R@1@{k}']:>{cw_k}.4f}" for k in args.ks))
    print("=" * len(hdr))

    # CSV
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = args.output or os.path.join(
        out_dir, f"reproduce_fig5_{args.dataset}_nb{args.nb}_nq{args.nq}.csv"
    )
    fieldnames = ["method", "dataset", "dim", "nb", "nq", "encode_s", "estimate_s",
                  *[f"R@1@{k}" for k in args.ks]]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({"dataset": args.dataset, "dim": d,
                        "nb": base.shape[0], "nq": queries.shape[0], **r})
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
