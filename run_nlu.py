"""CLI della Parte 2 (NLU): run | sweep | aggregate | finalize | export.

Gemello di `run.py` (LM) ma per il task intent+slot. Artefatti in `runs_nlu/`,
report con prefisso `partA_nlu_`. La run migliore e' quella con la **slot F1 di dev**
piu' alta.

Esempi:
  python run_nlu.py run      --config configs/nlu/experiments/_smoke.yaml
  python run_nlu.py sweep    --config configs/nlu/experiments/00_baseline.yaml
  python run_nlu.py aggregate
  python run_nlu.py finalize            # seleziona la run con dev slot F1 massima
  python run_nlu.py export              # genera NLU/part_A standalone
"""
import os
import json
import glob
import copy
import argparse

from nlu_pipeline.config import load_config, expand_sweep
from nlu_pipeline.experiment import run_experiment_nlu
from nlu_pipeline.aggregate import write_reports

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs_nlu")
REPORTS = os.path.join(ROOT, "reports")
DATASET = os.path.join(ROOT, "dataset", "ATIS")


def build_parser():
    p = argparse.ArgumentParser(description="Pipeline NLU (Parte 2) — CLI")
    sub = p.add_subparsers(dest="command", required=True)
    pr = sub.add_parser("run", help="Esegue una singola run dalla config YAML")
    pr.add_argument("--config", required=True)
    ps = sub.add_parser("sweep", help="Esegue il sweep completo (prodotto cartesiano)")
    ps.add_argument("--config", required=True)
    sub.add_parser("aggregate", help="Aggrega le run NLU in tabelle/grafici (partA_nlu_*)")
    pf = sub.add_parser("finalize", help="Finalizza la run migliore sul test set")
    pf.add_argument("--run", default=None)
    sub.add_parser("export", help="Esporta il pacchetto di consegna NLU/part_A")
    return p


def select_best_run(runs_root):
    """Ritorna le metrics della run con dev slot F1 MASSIMA (None ignorate)."""
    best = None
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))
        if m.get("best_dev_slot_f1") is None:
            continue
        if best is None or m["best_dev_slot_f1"] > best["best_dev_slot_f1"]:
            best = m
    return best


def _finalize(run_name=None):
    if run_name is None:
        best = select_best_run(RUNS)
        if best is None:
            print("Nessuna run trovata.")
            return
        cfg = best["config"]
        print(f"Run migliore: {best['name']} (dev slot F1={best['best_dev_slot_f1']:.2f})")
    else:
        cfg = json.load(open(os.path.join(RUNS, run_name, "metrics.json")))["config"]
        print(f"Run forzata: {run_name}")

    cfg = copy.deepcopy(cfg)
    cfg["mode"] = "final"
    cfg.setdefault("experiment", {})["name"] = cfg["experiment"]["name"] + "__FINAL"
    run_dir = run_experiment_nlu(cfg, RUNS, DATASET)
    m = json.load(open(os.path.join(run_dir, "metrics.json")))
    print(f"TEST  slot F1: {m['test_slot_f1']:.2f}  |  intent acc: {m['test_intent_acc']*100:.2f}%")


def main():
    args = build_parser().parse_args()
    if args.command == "run":
        run_experiment_nlu(load_config(args.config), RUNS, DATASET)
    elif args.command == "sweep":
        for cfg in expand_sweep(load_config(args.config)):
            run_experiment_nlu(cfg, RUNS, DATASET)
    elif args.command == "aggregate":
        write_reports(RUNS, REPORTS)
        print(f"Report (partA_nlu_*) scritti in {REPORTS}")
    elif args.command == "finalize":
        _finalize(args.run)
    elif args.command == "export":
        from nlu_pipeline_export import export_part_a_nlu
        export_part_a_nlu(ROOT)


if __name__ == "__main__":
    main()
