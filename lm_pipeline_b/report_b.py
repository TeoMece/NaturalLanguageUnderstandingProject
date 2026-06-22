"""Parte 1.B — report delle run LoRA (tabelle + grafico comparativo).

Riusa le funzioni PURE di `lm_pipeline.aggregate` (`collect_runs`, `to_latex`,
`to_markdown`), che lavorano su qualsiasi cartella di run con `metrics.json` dello
schema atteso. Scriviamo file con prefisso `partB_` per non sovrascrivere quelli
di 1.A. Non tocchiamo `aggregate.py` (cosi' i suoi test restano verdi).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lm_pipeline.aggregate import collect_runs, to_latex, to_markdown


def write_reports_b(runs_root, reports_dir):
    """Scrive partB_summary.tex/.md e figures/comparison_b.pdf. Ritorna le righe."""
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)
    rows = collect_runs(runs_root)

    with open(os.path.join(reports_dir, "partB_summary.tex"), "w") as f:
        f.write(to_latex(rows))
    with open(os.path.join(reports_dir, "partB_summary.md"), "w") as f:
        f.write(to_markdown(rows))

    plt.figure()
    plt.barh([r["name"] for r in rows], [r["best_valid_ppl"] for r in rows])
    plt.xlabel("valid PPL")
    plt.title("Parte 1.B — confronto run LoRA")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "figures", "comparison_b.pdf"))
    plt.close()
    return rows
