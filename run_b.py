"""CLI della Parte 1.B (LoRA): run | sweep | aggregate | finalize | export.

Gemello di `run.py` (1.A) ma per il fine-tuning LoRA. Gli artefatti vanno in
`runs_b/` (separati dalle run di 1.A in `runs/`), i report con prefisso `partB_`.

Esempi:
  python run_b.py run      --config configs/experiments_b/_smoke.yaml
  python run_b.py sweep    --config configs/experiments_b/00_baseline.yaml
  python run_b.py aggregate
  python run_b.py finalize            # seleziona automaticamente la run migliore
  python run_b.py finalize --run NOME # forza una run specifica
  python run_b.py export              # genera LM/part_B standalone
"""
import os
import json
import glob
import copy
import argparse

from lm_pipeline.config import load_config, expand_sweep
from lm_pipeline_b.experiment_lora import run_experiment_lora
from lm_pipeline_b.report_b import write_reports_b

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs_b")                  # run di B separate da quelle di A
REPORTS = os.path.join(ROOT, "reports")
DATASET = os.path.join(ROOT, "dataset", "PennTreeBank")


def build_parser():
    """Parser con gli stessi 5 subcommand di run.py, adattati a B."""
    p = argparse.ArgumentParser(description="Pipeline LM (Parte 1.B, LoRA) — CLI")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="Esegue una singola run LoRA dalla config YAML")
    pr.add_argument("--config", required=True, help="Path al file YAML dell'esperimento")

    ps = sub.add_parser("sweep", help="Esegue il sweep completo (prodotto cartesiano)")
    ps.add_argument("--config", required=True, help="Path al file YAML con la sezione 'sweep'")

    sub.add_parser("aggregate", help="Aggrega le run di B in tabelle/grafici (partB_*)")

    pf = sub.add_parser("finalize", help="Finalizza la run migliore sul test set")
    pf.add_argument("--run", default=None, help="Nome run da finalizzare (default: PPL minima)")

    sub.add_parser("export", help="Esporta il pacchetto di consegna LM/part_B")
    return p


def select_best_run(runs_root):
    """Ritorna le metrics della run con best_valid_ppl minima (None ignorate)."""
    best = None
    for path in glob.glob(os.path.join(runs_root, "*", "metrics.json")):
        m = json.load(open(path))
        if m.get("best_valid_ppl") is None:
            continue
        if best is None or m["best_valid_ppl"] < best["best_valid_ppl"]:
            best = m
    return best


def _finalize(run_name=None):
    """Ri-allena sul dataset completo in mode 'final' e stampa la PPL su test."""
    if run_name is None:
        best = select_best_run(RUNS)
        if best is None:
            print("Nessuna run trovata.")
            return
        cfg = best["config"]
        print(f"Run migliore: {best['name']} (valid PPL={best['best_valid_ppl']:.2f})")
    else:
        cfg = json.load(open(os.path.join(RUNS, run_name, "metrics.json")))["config"]
        print(f"Run forzata: {run_name}")

    cfg = copy.deepcopy(cfg)
    cfg["mode"] = "final"
    cfg.setdefault("experiment", {})["name"] = cfg["experiment"]["name"] + "__FINAL"

    run_dir = run_experiment_lora(cfg, RUNS, DATASET)
    m = json.load(open(os.path.join(run_dir, "metrics.json")))
    print(f"PPL finale su test: {m['test_ppl']:.2f}")


def main():
    args = build_parser().parse_args()
    if args.command == "run":
        run_experiment_lora(load_config(args.config), RUNS, DATASET)
    elif args.command == "sweep":
        for cfg in expand_sweep(load_config(args.config)):
            run_experiment_lora(cfg, RUNS, DATASET)
    elif args.command == "aggregate":
        write_reports_b(RUNS, REPORTS)
        print(f"Report (partB_*) scritti in {REPORTS}")
    elif args.command == "finalize":
        _finalize(args.run)
    elif args.command == "export":
        from lm_pipeline_b_export import export_part_b
        export_part_b(ROOT)


if __name__ == "__main__":
    main()
