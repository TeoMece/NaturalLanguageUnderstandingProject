"""Aggregatore multi-seed: media ± deviazione standard per configurazione.

Su corpora piccoli (ATIS) e per risultati affidabili, il lab raccomanda di ripetere
ogni run con seed diversi e riportare media ± std. Questo script raggruppa le run che
differiscono SOLO per il seed (nome run con suffisso `__seed<N>`) e calcola media/std
delle metriche presenti nel `metrics.json` (sia LM che NLU).

Workflow:
  1. nei config, aggiungi  `sweep: { experiment.seed: [42, 1, 2, 3, 4] }`  (anche in
     combinazione con altri sweep) -> ogni config genera N run `..._seed42`, `..._seed1`, ...
  2. lancia lo sweep (run.py/run_nlu.py/...).
  3. python seed_summary.py --runs <runs_dir>  -> tabella media ± std per config.

Uso:
  python seed_summary.py --runs runs_nlu
  python seed_summary.py --runs runs --out reports/seed_summary_lm.md
"""
import os
import re
import json
import glob
import argparse
from statistics import mean, stdev

# Metriche candidate: presenti a seconda della pipeline (LM vs NLU).
_METRICS = [
    "best_valid_ppl", "test_ppl",                 # LM
    "best_dev_slot_f1", "best_dev_intent_acc",     # NLU (dev)
    "test_slot_f1", "test_intent_acc",             # NLU (test)
]


def group_key(name):
    """Nome run senza il suffisso del seed -> chiave di raggruppamento.

    Es. 'partA_nlu_arch__d_model256__num_layers2__seed42' -> '...num_layers2'
        'partB_nlu__lr5e-05__seed1__gpt2'                  -> 'partB_nlu__lr5e-05__gpt2'
    """
    return re.sub(r"__seed[^_]+", "", name)


def collect(runs_root):
    """Raggruppa i metrics.json per config (seed rimosso)."""
    groups = {}
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))
        key = group_key(m.get("name", os.path.basename(os.path.dirname(path))))
        groups.setdefault(key, []).append(m)
    return groups


def summarize(groups):
    """Per ogni gruppo: (media, std, n) di ciascuna metrica presente. std=0 se n=1."""
    rows = []
    for key, runs in sorted(groups.items()):
        row = {"config": key, "n_seeds": len(runs), "metrics": {}}
        for met in _METRICS:
            vals = [r[met] for r in runs if r.get(met) is not None]
            if vals:
                row["metrics"][met] = (mean(vals), stdev(vals) if len(vals) > 1 else 0.0, len(vals))
        rows.append(row)
    return rows


def _fmt(metric, mu, sd):
    # PPL: valore grezzo; F1/accuracy (0-1): mostrati in % per leggibilita'
    if "ppl" in metric:
        return f"{mu:.2f} ± {sd:.2f}"
    return f"{mu * 100:.2f} ± {sd * 100:.2f}"


def to_markdown(rows):
    mets = [m for m in _METRICS if any(m in r["metrics"] for r in rows)]
    head = "| config | n | " + " | ".join(mets) + " |\n"
    head += "|" + "---|" * (len(mets) + 2) + "\n"
    body = ""
    for r in rows:
        cells = [(_fmt(m, *r["metrics"][m]) if m in r["metrics"] else "-") for m in mets]
        body += f"| {r['config']} | {r['n_seeds']} | " + " | ".join(cells) + " |\n"
    return head + body


def main():
    ap = argparse.ArgumentParser(description="Media ± std multi-seed per configurazione.")
    ap.add_argument("--runs", required=True, help="cartella delle run (es. runs_nlu)")
    ap.add_argument("--out", default=None, help="file .md di output (opzionale)")
    args = ap.parse_args()

    groups = collect(args.runs)
    if not groups:
        print(f"Nessuna run con metrics.json in '{args.runs}/'.")
        return
    md = to_markdown(summarize(groups))
    print(md)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md)
        print(f"Scritto: {args.out}")


if __name__ == "__main__":
    main()
