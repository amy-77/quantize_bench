#!/usr/bin/env python3
"""Plot Fig. 5(a)-style Recall@1@k curves from a reproduce CSV.

Reads result/reproduce_fig5_<dataset>_nb<...>_nq<...>.csv and produces a PNG
with one curve per method, x-axis top-k on log2 scale, y-axis Recall@1@k.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Visual style matching paper Fig. 5: solid lines for 4-bit, dashed for 2-bit;
# distinct color per method family.
STYLE = {
    "TQ-MSE-2bit":  {"color": "#1f77b4", "linestyle": "--", "marker": "o", "label": "TurboQuant-MSE 2 bits"},
    "TQ-MSE-4bit":  {"color": "#1f77b4", "linestyle": "-",  "marker": "o", "label": "TurboQuant-MSE 4 bits"},
    "TQ-Prod-2bit": {"color": "#d62728", "linestyle": "--", "marker": "s", "label": "TurboQuant-Prod 2 bits"},
    "TQ-Prod-4bit": {"color": "#d62728", "linestyle": "-",  "marker": "s", "label": "TurboQuant-Prod 4 bits"},
}


def load_csv(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def extract_recall_columns(rows):
    if not rows:
        return []
    ks = []
    for k in rows[0]:
        if k.startswith("R@1@"):
            ks.append(int(k.split("@")[-1]))
    return sorted(ks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="result/reproduce_fig5_glove_nb100000_nq10000.csv")
    p.add_argument("--out", default="result/reproduce_fig5_glove.png")
    p.add_argument("--title", default="GloVe - d=200")
    p.add_argument("--methods", nargs="+", default=None,
                   help="Subset of method names to plot (default: all that match STYLE)")
    p.add_argument("--ymin", type=float, default=None)
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(here, args.csv)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)

    rows = load_csv(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")
    ks = extract_recall_columns(rows)
    if not ks:
        raise SystemExit(f"No R@1@k columns in {csv_path}")

    rows_by_method = {r["method"]: r for r in rows}
    methods = args.methods or [m for m in STYLE if m in rows_by_method]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ymin_seen = 1.0
    for m in methods:
        if m not in rows_by_method:
            print(f"warning: '{m}' not in csv, skipping")
            continue
        r = rows_by_method[m]
        ys = [float(r[f"R@1@{k}"]) for k in ks]
        ymin_seen = min(ymin_seen, min(ys))
        st = STYLE.get(m, {"label": m})
        ax.plot(ks, ys, **st)

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Top-k")
    ax.set_ylabel("Recall@1@k")
    ax.set_title(args.title)

    ymin = args.ymin if args.ymin is not None else max(0.0, round(ymin_seen - 0.05, 1))
    ax.set_ylim(ymin, 1.005)

    ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
