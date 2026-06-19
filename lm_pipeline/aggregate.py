"""Aggregazione delle run in tabelle (LaTeX/Markdown) e plot comparativo.

Scansiona la cartella runs/, legge ogni metrics.json e produce materiale pronto
per il report: una tabella ordinata per perplexity di validazione crescente e
una figura a barre orizzontali per confronto visivo immediato.

Schema atteso di metrics.json
------------------------------
{
  "name":           str,          # identificatore della run
  "best_valid_ppl": float,        # perplexity migliore su validation set
  "test_ppl":       float | null, # perplexity su test set (None se non ancora calcolata)
  "n_params":       int,          # numero di parametri del modello
  "seconds":        float,        # tempo totale di addestramento in secondi
  "config": {
    "optim": {
      "lr": float               # learning rate usato
    }
  }
}
"""

import os
import json
import glob

# Usare il backend "Agg" (non interattivo) PRIMA di importare pyplot,
# per evitare errori in ambienti senza display (server, CI, test).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Raccolta e ordinamento dei risultati
# ---------------------------------------------------------------------------

def collect_runs(runs_root):
    """Legge tutti i metrics.json sotto runs_root e li ordina per best_valid_ppl.

    Parameters
    ----------
    runs_root : str
        Percorso alla cartella radice delle run (es. "runs/").
        Ogni sotto-cartella deve contenere un file metrics.json.

    Returns
    -------
    list[dict]
        Lista di dizionari con le metriche di ogni run, ordinata per
        best_valid_ppl crescente (la run migliore è in posizione 0).
        Le run con best_valid_ppl == None vengono spostate in fondo.
    """
    rows = []

    # glob cerca il pattern <runs_root>/<qualsiasi cartella>/metrics.json
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))

        # Estrae i campi di interesse; usa fallback sicuri per campi mancanti
        rows.append({
            "name": m.get("name", os.path.basename(os.path.dirname(path))),
            # Learning rate annidato in config.optim.lr
            "lr": m.get("config", {}).get("optim", {}).get("lr"),
            "best_valid_ppl": m.get("best_valid_ppl"),
            "test_ppl": m.get("test_ppl"),
            "n_params": m.get("n_params"),
            "seconds": m.get("seconds"),
            # Conserva il percorso della cartella per usi successivi (es. caricare il checkpoint)
            "_dir": os.path.dirname(path),
        })

    # Ordina per PPL crescente; le run senza PPL (None) vengono messe in coda
    # grazie al primo elemento della tupla (True > False in Python).
    rows.sort(key=lambda r: (r["best_valid_ppl"] is None, r["best_valid_ppl"]))
    return rows


# ---------------------------------------------------------------------------
# Esportazione in Markdown
# ---------------------------------------------------------------------------

def to_markdown(rows):
    """Restituisce una stringa con la tabella Markdown delle run.

    Formato
    -------
    | name | lr | valid PPL | test PPL | #params | s |
    |---|---|---|---|---|---|
    | run_A | 0.001 | 123.45 | - | 1000000 | 300.0 |
    ...

    Parameters
    ----------
    rows : list[dict]
        Output di collect_runs().

    Returns
    -------
    str
        Tabella Markdown pronta per essere incollata in un README o report.
    """
    # Intestazione e riga separatrice
    head = "| name | lr | valid PPL | test PPL | #params | s |\n"
    head += "|---|---|---|---|---|---|\n"

    body = ""
    for r in rows:
        # test_ppl può essere None se il test set non è ancora stato valutato
        test_str = str(r["test_ppl"]) if r["test_ppl"] is not None else "-"
        body += (
            f"| {r['name']} | {r['lr']} | {r['best_valid_ppl']:.2f} | "
            f"{test_str} | "
            f"{r['n_params']} | {r['seconds']} |\n"
        )

    return head + body


# ---------------------------------------------------------------------------
# Esportazione in LaTeX (booktabs)
# ---------------------------------------------------------------------------

def to_latex(rows):
    """Restituisce una stringa con la tabella LaTeX (booktabs) delle run.

    Usa i comandi \\toprule, \\midrule, \\bottomrule del pacchetto booktabs
    per una tipografia professionale secondo lo stile IEEE.

    Parameters
    ----------
    rows : list[dict]
        Output di collect_runs().

    Returns
    -------
    str
        Ambiente tabular LaTeX pronto da incollare nel report.
    """
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"esperimento & lr & valid PPL & test PPL & \#params & s \\",
        r"\midrule",
    ]

    for r in rows:
        # test_ppl formattato come float con 2 decimali, oppure trattino
        test = f"{r['test_ppl']:.2f}" if r["test_ppl"] is not None else "-"
        # In LaTeX il carattere '_' deve essere escapato come '\_'
        name = r["name"].replace("_", r"\_")
        lines.append(
            f"{name} & {r['lr']} & {r['best_valid_ppl']:.2f} & {test} & "
            f"{r['n_params']} & {r['seconds']} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scrittura dei report su disco (entry-point principale)
# ---------------------------------------------------------------------------

def write_reports(runs_root, reports_dir):
    """Scrive i file di report nella cartella reports_dir.

    Genera:
    - partA_summary.tex   : tabella LaTeX (booktabs)
    - partA_summary.md    : tabella Markdown
    - figures/comparison.pdf : barre orizzontali con la valid PPL di ogni run

    Parameters
    ----------
    runs_root : str
        Cartella radice delle run (es. "runs/").
    reports_dir : str
        Cartella di destinazione per i report (es. "reports/").

    Returns
    -------
    list[dict]
        Le stesse righe restituite da collect_runs(), per uso programmatico.
    """
    # Assicura che la cartella delle figure esista
    os.makedirs(os.path.join(reports_dir, "figures"), exist_ok=True)

    # Raccoglie e ordina le run
    rows = collect_runs(runs_root)

    # --- Tabella LaTeX ---
    with open(os.path.join(reports_dir, "partA_summary.tex"), "w") as f:
        f.write(to_latex(rows))

    # --- Tabella Markdown ---
    with open(os.path.join(reports_dir, "partA_summary.md"), "w") as f:
        f.write(to_markdown(rows))

    # --- Figura comparativa (barre orizzontali, PPL per run) ---
    plt.figure()
    names = [r["name"] for r in rows]
    ppls = [r["best_valid_ppl"] for r in rows]
    plt.barh(names, ppls)           # barre orizzontali: facile da leggere con molte run
    plt.xlabel("valid PPL")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, "figures", "comparison.pdf"))
    plt.close()                     # libera la memoria della figura

    return rows
