"""Test per lm_pipeline.experiment.

Copre la logica di resumabilita' (already_done): verifica che una run il cui
`metrics.json` e' gia' presente venga riconosciuta come completata, e che una
run inesistente non lo sia. Il test e2e di run_experiment e' coperto altrove
(smoke test) poiche' richiede tokenizer e qualche secondo di training.
"""
import os
import json
from lm_pipeline.experiment import run_experiment, already_done


def test_already_done_detects_completed_run(tmp_path):
    """already_done ritorna True se metrics.json esiste, False altrimenti."""
    # Crea la struttura di una run completata: runs/exp/metrics.json
    rd = tmp_path / "runs" / "exp"
    rd.mkdir(parents=True)
    (rd / "metrics.json").write_text(json.dumps({"best_ppl": 10}))

    # La run 'exp' ha gia' un metrics.json -> deve essere considerata completata
    assert already_done(str(tmp_path / "runs"), "exp") is True

    # La run 'altro' non esiste affatto -> non completata
    assert already_done(str(tmp_path / "runs"), "altro") is False
