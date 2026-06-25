"""Test dell'aggregatore multi-seed: raggruppamento per config (seed rimosso) + media/std."""
import os
import json

from seed_summary import group_key, collect, summarize


def test_group_key_strips_seed():
    assert group_key("partA_nlu_baseline__seed42") == "partA_nlu_baseline"
    assert group_key("partA_arch__d_model256__num_layers2__seed1") == "partA_arch__d_model256__num_layers2"
    assert group_key("partB_nlu__lr5e-05__seed3__gpt2") == "partB_nlu__lr5e-05__gpt2"


def _write_run(root, name, metrics):
    d = os.path.join(root, name)
    os.makedirs(d)
    json.dump({"name": name, **metrics}, open(os.path.join(d, "metrics.json"), "w"))


def test_collect_and_summarize(tmp_path):
    root = str(tmp_path)
    # due seed della stessa config (NLU) + una config diversa
    _write_run(root, "cfgA__seed42", {"best_dev_slot_f1": 0.96, "best_dev_intent_acc": 0.98})
    _write_run(root, "cfgA__seed1", {"best_dev_slot_f1": 0.94, "best_dev_intent_acc": 0.96})
    _write_run(root, "cfgB__seed42", {"best_dev_slot_f1": 0.90, "best_dev_intent_acc": 0.92})

    groups = collect(root)
    assert set(groups) == {"cfgA", "cfgB"}
    assert len(groups["cfgA"]) == 2 and len(groups["cfgB"]) == 1

    rows = {r["config"]: r for r in summarize(groups)}
    muA, sdA, nA = rows["cfgA"]["metrics"]["best_dev_slot_f1"]
    assert abs(muA - 0.95) < 1e-9 and nA == 2 and sdA > 0
    # n=1 -> std 0
    _, sdB, nB = rows["cfgB"]["metrics"]["best_dev_slot_f1"]
    assert nB == 1 and sdB == 0.0
