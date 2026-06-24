"""Aggregazione delle run NLU in tabelle (LaTeX/Markdown) e grafico comparativo.

Gemello di `lm_pipeline.aggregate`, con le metriche del task NLU: slot F1 e intent
accuracy (dev e test). Ordina per slot F1 di dev DECRESCENTE (piu' alta = migliore).
"""
import os
import json
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_runs(runs_root):
    """Legge tutti i metrics.json sotto runs_root, ordinati per dev slot F1 decrescente."""
    rows = []
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))
        rows.append({
            "name": m.get("name", os.path.basename(os.path.dirname(path))),
            "lr": m.get("config", {}).get("optim", {}).get("lr"),
            "dev_slot_f1": m.get("best_dev_slot_f1"),
            "dev_intent_acc": m.get("best_dev_intent_acc"),
            "test_slot_f1": m.get("test_slot_f1"),
            "test_intent_acc": m.get("test_intent_acc"),
            "n_params": m.get("n_params"),
            "_dir": os.path.dirname(path),
        })
    # ordina per dev slot F1 decrescente; None in fondo
    rows.sort(key=lambda r: (r["dev_slot_f1"] is None, -(r["dev_slot_f1"] or 0)))
    return rows


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v*100:.2f}" if pct else f"{v:.2f}"


def to_markdown(rows):
    """Tabella Markdown: name | lr | dev F1 | dev acc | test F1 | test acc | #params."""
    head = "| name | lr | dev slot F1 | dev intent acc | test slot F1 | test intent acc | #params |\n"
    head += "|---|---|---|---|---|---|---|\n"
    body = ""
    for r in rows:
        body += (f"| {r['name']} | {r['lr']} | {_fmt(r['dev_slot_f1'], pct=True)} | "
                 f"{_fmt(r['dev_intent_acc'], pct=True)} | {_fmt(r['test_slot_f1'], pct=True)} | "
                 f"{_fmt(r['test_intent_acc'], pct=True)} | {r['n_params']} |\n")
    return head + body


def to_latex(rows):
    """Tabella LaTeX (booktabs) con le stesse colonne."""
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"esperimento & lr & dev slot F1 & dev intent acc & test slot F1 & test intent acc & \#params \\",
             r"\midrule"]
    for r in rows:
        name = r["name"].replace("_", r"\_")
        lines.append(f"{name} & {r['lr']} & {_fmt(r['dev_slot_f1'], pct=True)} & "
                     f"{_fmt(r['dev_intent_acc'], pct=True)} & {_fmt(r['test_slot_f1'], pct=True)} & "
                     f"{_fmt(r['test_intent_acc'], pct=True)} & {r['n_params']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write_reports(runs_root, reports_dir):
    """Scrive partB_nlu_summary.tex/.md e figures/comparison_nlu_b.pdf. Ritorna le righe."""
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)
    rows = collect_runs(runs_root)
    with open(os.path.join(reports_dir, "partB_nlu_summary.tex"), "w") as f:
        f.write(to_latex(rows))
    with open(os.path.join(reports_dir, "partB_nlu_summary.md"), "w") as f:
        f.write(to_markdown(rows))

    plt.figure()
    plt.barh([r["name"] for r in rows], [100 * (r["dev_slot_f1"] or 0) for r in rows])
    plt.xlabel("dev slot F1 (%)")
    plt.title("Parte 2.B — confronto run (slot F1 di dev)")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "figures", "comparison_nlu_b.pdf"))
    plt.close()
    return rows
