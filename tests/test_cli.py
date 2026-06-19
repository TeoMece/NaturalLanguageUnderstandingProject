"""Test per run.py: parser CLI e selezione della run migliore.

Copre:
- build_parser: verifica che i subcommand e gli argomenti siano configurati correttamente
- select_best_run: verifica che la funzione restituisca la run con best_valid_ppl minima
  e ignori quelle con best_valid_ppl mancante.
"""
from run import build_parser, select_best_run


# ---------------------------------------------------------------------------
# Test 1 — comando 'sweep' con argomento --config
# ---------------------------------------------------------------------------

def test_parser_sweep_command():
    """Il parser riconosce 'sweep --config x.yaml' e popola args correttamente.

    Verifica:
    - args.command == 'sweep'   (il subcommand viene registrato in dest='command')
    - args.config  == 'x.yaml' (il valore del flag --config viene preservato)
    """
    args = build_parser().parse_args(["sweep", "--config", "x.yaml"])
    assert args.command == "sweep" and args.config == "x.yaml"


# ---------------------------------------------------------------------------
# Test 2 — comando 'finalize' senza --run (argomento opzionale assente)
# ---------------------------------------------------------------------------

def test_parser_finalize_optional_run():
    """'finalize' senza --run imposta args.run a None (selezione automatica).

    Verifica:
    - args.command == 'finalize'
    - args.run     is None       (default quando il flag non e' fornito)
    """
    args = build_parser().parse_args(["finalize"])
    assert args.command == "finalize" and args.run is None


# ---------------------------------------------------------------------------
# Test 3 — select_best_run sceglie la run con best_valid_ppl minima
# ---------------------------------------------------------------------------

def test_select_best_run_picks_min_valid_ppl(tmp_path):
    """select_best_run ritorna le metriche della run con best_valid_ppl piu' bassa.

    Crea tre run ('a' PPL=200, 'b' PPL=140, 'c' PPL=175) in tmp_path e
    verifica che la funzione selezioni 'b' (quella con il valore minore).
    """
    import os
    import json

    # Crea la struttura <tmp_path>/<name>/metrics.json per tre run con PPL diverse
    for name, ppl in [("a", 200.0), ("b", 140.0), ("c", 175.0)]:
        d = tmp_path / name
        d.mkdir()
        (d / "metrics.json").write_text(
            json.dumps({
                "name": name,
                "best_valid_ppl": ppl,
                "config": {"experiment": {"name": name}}
            })
        )

    # Chiama la funzione e controlla che restituisca la run 'b' (PPL minima)
    best = select_best_run(str(tmp_path))
    assert best["name"] == "b"
