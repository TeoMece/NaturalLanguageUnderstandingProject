import json
import torch
from lm_pipeline.tracking import set_seed, make_run_dir, save_metrics, save_curves

def test_set_seed_makes_torch_deterministic():
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)

def test_make_run_dir_creates_unique_dir(tmp_path):
    d1 = make_run_dir(str(tmp_path), "exp")
    d2 = make_run_dir(str(tmp_path), "exp")
    assert d1 != d2                      # nomi collidenti -> suffisso incrementale
    import os
    assert os.path.isdir(d1) and os.path.isdir(d2)

def test_save_metrics_writes_json(tmp_path):
    path = save_metrics(str(tmp_path), {"best_ppl": 123.4, "device": "cpu"})
    data = json.load(open(path))
    assert data["best_ppl"] == 123.4

def test_save_curves_writes_csv_and_pdf(tmp_path):
    hist = {"train_loss": [2.0, 1.5], "valid_loss": [2.1, 1.6], "valid_ppl": [8.0, 5.0]}
    csv_path, pdf_path = save_curves(str(tmp_path), hist)
    import os
    assert os.path.exists(csv_path) and os.path.exists(pdf_path)
    assert "valid_ppl" in open(csv_path).readline()
