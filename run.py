"""CLI unico della pipeline LM: run | sweep | aggregate | finalize | export.

Questo modulo e' il punto di ingresso principale del progetto. Ogni subcommand
delega il lavoro ai moduli interni di `lm_pipeline`, mantenendo questo file
puramente orchestrativo.

Esempi di utilizzo:
  python run.py run      --config configs/experiments/00_baseline.yaml
  python run.py sweep    --config configs/experiments/00_baseline.yaml
  python run.py aggregate
  python run.py finalize            # seleziona automaticamente la run migliore
  python run.py finalize --run NOME # forza una run specifica per la finalizzazione
  python run.py export              # richiede Task 9 (lm_pipeline_export)

Struttura del progetto:
  run.py                <- questo file (CLI entry-point)
  lm_pipeline/          <- package con tutta la logica (config, data, model, ...)
  configs/experiments/  <- file YAML delle configurazioni
  runs/                 <- artefatti delle run (creati da run_experiment)
  reports/              <- tabelle e figure aggregate (create da write_reports)
  dataset/PennTreeBank/ <- corpus PTB (train/valid/test.txt)
"""
import os
import json
import glob
import copy
import argparse

# Importa l'API pubblica del package lm_pipeline.
# Questi import vengono eseguiti all'avvio del modulo, PRIMA che argparse
# analizzi i flag: e' accettabile perche' lm_pipeline e' sempre disponibile.
from lm_pipeline.config import load_config, expand_sweep
from lm_pipeline.experiment import run_experiment
from lm_pipeline.aggregate import write_reports

# ---------------------------------------------------------------------------
# Path costanti derivati dalla posizione di run.py
# ---------------------------------------------------------------------------

# Cartella radice del progetto (dove si trova questo file).
# Usare __file__ rende run.py spostabile senza rompere i path relativi.
ROOT = os.path.dirname(os.path.abspath(__file__))

# Cartella radice di tutte le run (creata automaticamente da run_experiment).
RUNS = os.path.join(ROOT, "runs")

# Cartella di destinazione per i report aggregati (creata da write_reports).
REPORTS = os.path.join(ROOT, "reports")

# Cartella del corpus PennTreeBank (deve contenere ptb.train/valid/test.txt).
DATASET = os.path.join(ROOT, "dataset", "PennTreeBank")


# ---------------------------------------------------------------------------
# Parser CLI
# ---------------------------------------------------------------------------

def build_parser():
    """Costruisce e restituisce il parser argparse con i 5 subcommand.

    Subcommand disponibili:
    - run      --config PATH  : esegue una singola run dalla config specificata
    - sweep    --config PATH  : esegue il prodotto cartesiano (sweep) di una config
    - aggregate               : raccoglie tutte le run e scrive i report
    - finalize [--run NOME]   : finalizza la run migliore (o quella specificata)
    - export                  : esporta il pacchetto di consegna (Task 9)

    L'argomento `--config` e' obbligatorio per 'run' e 'sweep', mentre
    `--run` e' opzionale per 'finalize' (default None = selezione automatica).

    Returns
    -------
    argparse.ArgumentParser
        Parser configurato, pronto per chiamare .parse_args().
    """
    p = argparse.ArgumentParser(
        description="Pipeline LM (Parte 1.A) — entry-point CLI"
    )
    # dest='command' registra il subcommand scelto in args.command
    sub = p.add_subparsers(dest="command", required=True)

    # -- run: esegue una singola run --
    pr = sub.add_parser("run", help="Esegue una singola run dalla config YAML")
    pr.add_argument("--config", required=True,
                    help="Path al file YAML dell'esperimento")

    # -- sweep: prodotto cartesiano di tutti i valori nella sezione 'sweep' --
    ps = sub.add_parser("sweep",
                         help="Esegue il sweep completo (prodotto cartesiano)")
    ps.add_argument("--config", required=True,
                    help="Path al file YAML con la sezione 'sweep'")

    # -- aggregate: legge tutte le run e produce i report --
    sub.add_parser("aggregate",
                   help="Aggrega le run in tabelle LaTeX/Markdown e grafici")

    # -- finalize: re-run in mode:final sul full dataset --
    pf = sub.add_parser(
        "finalize",
        help="Finalizza la run migliore (o quella specificata) sul test set"
    )
    pf.add_argument(
        "--run", default=None,
        help="Nome della run da finalizzare (default: la run con PPL minima)"
    )

    # -- export: pacchetto di consegna (richiede Task 9) --
    sub.add_parser("export",
                   help="Esporta il pacchetto di consegna (Task 9)")

    return p


# ---------------------------------------------------------------------------
# Selezione della run migliore
# ---------------------------------------------------------------------------

def select_best_run(runs_root):
    """Ritorna le metrics della run con best_valid_ppl minima.

    Scansiona tutti i metrics.json direttamente sotto runs_root e trova
    la run con il valore di best_valid_ppl piu' basso (migliore).
    Le run prive di best_valid_ppl (campo None o assente) vengono ignorate.

    Parameters
    ----------
    runs_root : str
        Percorso alla cartella radice delle run (es. "runs/").
        Ogni sotto-cartella deve contenere un metrics.json con almeno
        il campo "best_valid_ppl".

    Returns
    -------
    dict or None
        Dizionario con le metriche della run migliore, oppure None se
        non viene trovata nessuna run valida.
    """
    best = None  # tiene traccia della run migliore trovata finora

    # Itera su tutti i metrics.json delle sotto-cartelle dirette di runs_root
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))

        # Salta le run incomplete o senza best_valid_ppl valido
        if m.get("best_valid_ppl") is None:
            continue

        # Aggiorna il best se questa run ha PPL piu' bassa (migliore)
        if best is None or m["best_valid_ppl"] < best["best_valid_ppl"]:
            best = m

    return best


# ---------------------------------------------------------------------------
# Logica di finalizzazione
# ---------------------------------------------------------------------------

def _finalize(run_name=None):
    """Esegue la run finale sul dataset completo e stampa la PPL su test.

    Se run_name e' None, seleziona automaticamente la run con best_valid_ppl
    minima tra quelle gia' completate (sotto RUNS/). Se run_name e' fornito,
    carica direttamente la config di quella run.

    In entrambi i casi, la config viene copiata e modificata:
    - mode viene impostato a 'final' (attiva la valutazione su test)
    - il nome dell'esperimento viene decorato con '__FINAL' per distinguerlo
      dalla run originale (e per evitare che il meccanismo di resumabilita'
      salti la run finale).

    Parameters
    ----------
    run_name : str or None
        Nome della run da finalizzare. Se None, viene selezionata
        automaticamente la migliore.
    """
    if run_name is None:
        # Selezione automatica: cerca la run con PPL minima
        best = select_best_run(RUNS)
        if best is None:
            # Nessuna run trovata: avvisa l'utente e termina senza errori
            print("Nessuna run trovata.")
            return
        cfg = best["config"]
        print(f"Run migliore: {best['name']} (valid PPL={best['best_valid_ppl']:.2f})")
    else:
        # Modalita' manuale: carica la config dalla run specificata
        metrics_path = os.path.join(RUNS, run_name, "metrics.json")
        cfg = json.load(open(metrics_path))["config"]
        print(f"Run forzata: {run_name}")

    # Copia profonda per non modificare la config originale in memoria
    cfg = copy.deepcopy(cfg)

    # Imposta mode='final': run_experiment valutera' sul test set
    # e usera' il dataset completo (fraction=1.0)
    cfg["mode"] = "final"

    # Rinomina la run aggiungendo '__FINAL' per:
    # 1. distinguerla dalla run originale nell'output
    # 2. permettere la re-esecuzione (make_run_dir crea una cartella nuova)
    cfg.setdefault("experiment", {})["name"] = (
        cfg["experiment"]["name"] + "__FINAL"
    )

    # Esegue la run finale e legge il metrics.json prodotto
    run_dir = run_experiment(cfg, RUNS, DATASET)
    m = json.load(open(os.path.join(run_dir, "metrics.json")))

    # Stampa la PPL sul test set (la metrica definitiva del progetto)
    print(f"PPL finale su test: {m['test_ppl']:.2f}")


# ---------------------------------------------------------------------------
# Entry-point principale
# ---------------------------------------------------------------------------

def main():
    """Parsing degli argomenti e dispatch al subcommand corretto.

    Il flusso e':
    1. build_parser() crea il parser con i 5 subcommand
    2. parse_args() analizza sys.argv e popola args.command (e gli altri flag)
    3. Il blocco if/elif smista all'implementazione corretta

    Il branch 'export' importa lm_pipeline_export in modo lazy (dentro l'if)
    per evitare ImportError prima che Task 9 sia implementato: fino ad allora,
    tutti gli altri subcommand funzionano normalmente.
    """
    args = build_parser().parse_args()

    if args.command == "run":
        # Carica la config e avvia una singola run
        run_experiment(load_config(args.config), RUNS, DATASET)

    elif args.command == "sweep":
        # Espande la sezione 'sweep' nel prodotto cartesiano e avvia ogni run
        for cfg in expand_sweep(load_config(args.config)):
            run_experiment(cfg, RUNS, DATASET)

    elif args.command == "aggregate":
        # Raccoglie tutte le run e scrive i report (LaTeX, Markdown, PDF)
        write_reports(RUNS, REPORTS)
        print(f"Report scritti in {REPORTS}")

    elif args.command == "finalize":
        # Finalizza: seleziona la run migliore (o quella indicata) e valuta su test
        _finalize(args.run)

    elif args.command == "export":
        # Import lazy: lm_pipeline_export non esiste ancora (Task 9).
        # Importarlo qui evita che l'import a livello di modulo blocchi
        # gli altri subcommand prima che Task 9 sia completato.
        from lm_pipeline_export import export_part_a   # vedi Task 9
        export_part_a(ROOT)


# Eseguito solo se run.py viene lanciato direttamente (non importato come modulo)
if __name__ == "__main__":
    main()
