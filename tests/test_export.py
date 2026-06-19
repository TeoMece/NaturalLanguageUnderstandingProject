import os, subprocess, sys
from lm_pipeline_export import export_part_a

def test_export_creates_standalone_files(tmp_path):
    # prepara una finta struttura sorgente minima
    root = tmp_path
    (root / "lm_pipeline").mkdir()
    (root / "lm_pipeline" / "model.py").write_text("x = 'model'\n")
    (root / "lm_pipeline" / "data.py").write_text("y = 'data'\n")
    (root / "lm_pipeline" / "train.py").write_text("z = 'train'\n")
    out = root / "LM" / "part_A"
    export_part_a(str(root), out_dir=str(out), best_cfg={"model": {"d_model": 32}},
                  best_ckpt=None)
    for fname in ["model.py", "utils.py", "functions.py", "main.py"]:
        assert os.path.exists(os.path.join(str(out), fname))
    # i file copiati non devono importare lo strato di orchestrazione
    content = open(os.path.join(str(out), "model.py")).read()
    assert "config" not in content and "experiment" not in content
