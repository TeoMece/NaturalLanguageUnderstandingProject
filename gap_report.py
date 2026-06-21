"""Report dell'overfitting gap per ogni run.

Per ciascuna run legge curves.csv, individua l'epoca con la valid PPL migliore
e calcola il gap di generalizzazione (valid_ppl - train_ppl) a quell'epoca.
Serve a capire QUALI run overfittavano (gap grande) prima di decidere se il
dropout / altre regolarizzazioni hanno senso, o se conviene toccare le epoche.

train_ppl = exp(train_loss); valid_ppl e' gia' salvata in curves.csv.

Uso:
  python gap_report.py                         # legge runs/, scrive reports/overfitting_gaps.csv
  python gap_report.py --runs runs --out reports/gaps.csv
"""
import os
import csv
import glob
import math
import argparse


def gap_for_run(curves_path):
    """Ritorna (best_epoch, epochs_tot, train_ppl, valid_ppl, gap) o None se vuoto."""
    with open(curves_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    # Epoca con la valid PPL minima: e' il punto che l'early stopping seleziona.
    best = min(rows, key=lambda r: float(r["valid_ppl"]))
    train_ppl = math.exp(float(best["train_loss"]))
    valid_ppl = float(best["valid_ppl"])
    return (int(best["epoch"]), len(rows), train_ppl, valid_ppl, valid_ppl - train_ppl)


def main():
    ap = argparse.ArgumentParser(description="Overfitting gap per run.")
    ap.add_argument("--runs", default="runs", help="cartella delle run (default: runs)")
    ap.add_argument("--out", default="reports/overfitting_gaps.csv",
                    help="file CSV di output (default: reports/overfitting_gaps.csv)")
    args = ap.parse_args()

    records = []
    for curves in glob.glob(os.path.join(args.runs, "*", "curves.csv")):
        name = os.path.basename(os.path.dirname(curves))
        res = gap_for_run(curves)
        if res is None:
            print(f"  (saltata, curves.csv vuoto) {name}")
            continue
        best_epoch, epochs_tot, train_ppl, valid_ppl, gap = res
        records.append({
            "run": name,
            "best_epoch": best_epoch,
            "epochs_run": epochs_tot,
            "train_ppl": round(train_ppl, 2),
            "valid_ppl": round(valid_ppl, 2),
            "gap": round(gap, 2),
        })

    if not records:
        print(f"Nessuna run con curves.csv trovata in '{args.runs}/'.")
        return

    # Ordina per gap decrescente: in cima chi overfitta di piu'.
    records.sort(key=lambda r: r["gap"], reverse=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "best_epoch", "epochs_run",
                                          "train_ppl", "valid_ppl", "gap"])
        w.writeheader()
        w.writerows(records)

    # Stampa anche a schermo una tabellina leggibile.
    print(f"\n{'run':38s} {'best_ep':>7} {'epochs':>6} {'train_ppl':>10} {'valid_ppl':>10} {'gap':>8}")
    print("-" * 84)
    for r in records:
        print(f"{r['run']:38s} {r['best_epoch']:>7} {r['epochs_run']:>6} "
              f"{r['train_ppl']:>10.2f} {r['valid_ppl']:>10.2f} {r['gap']:>8.2f}")
    print(f"\nScritto: {args.out}  ({len(records)} run, ordinate per gap decrescente)")


if __name__ == "__main__":
    main()
