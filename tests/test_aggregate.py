"""Test suite per lm_pipeline/aggregate.py (Task 7).

Segue TDD: i test vengono scritti prima dell'implementazione.
Verifica: collect_runs ordina per PPL crescente, to_latex produce un ambiente
tabular con i valori numerici, to_markdown produce una riga di intestazione
riconoscibile.
"""
import os
import json
from lm_pipeline.aggregate import collect_runs, to_latex, to_markdown


def _make_run(root, name, ppl):
    """Helper: crea una sotto-cartella <name> con un metrics.json minimale."""
    d = os.path.join(root, name)
    os.makedirs(d)
    json.dump(
        {
            "name": name,
            "best_valid_ppl": ppl,
            "test_ppl": None,
            "n_params": 1000,
            "seconds": 12.0,
            "config": {"optim": {"lr": 0.001}},
        },
        open(os.path.join(d, "metrics.json"), "w"),
    )


def test_collect_runs_sorted_by_ppl(tmp_path):
    """collect_runs deve restituire le run ordinate per best_valid_ppl crescente."""
    _make_run(str(tmp_path), "a", 200.0)
    _make_run(str(tmp_path), "b", 150.0)
    rows = collect_runs(str(tmp_path))
    assert [r["name"] for r in rows] == ["b", "a"]  # ordinate per PPL crescente


def test_to_latex_contains_tabular_and_values(tmp_path):
    """to_latex deve produrre un ambiente {tabular} contenente i valori numerici."""
    _make_run(str(tmp_path), "a", 199.5)
    rows = collect_runs(str(tmp_path))
    tex = to_latex(rows)
    assert "tabular" in tex and "199.5" in tex


def test_to_markdown_has_header(tmp_path):
    """to_markdown deve produrre un'intestazione di tabella riconoscibile."""
    _make_run(str(tmp_path), "a", 199.5)
    rows = collect_runs(str(tmp_path))
    md = to_markdown(rows)
    assert "| name |" in md.lower() or "name" in md
