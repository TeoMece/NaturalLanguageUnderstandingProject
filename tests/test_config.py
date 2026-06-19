import textwrap
from lm_pipeline.config import load_config, set_dotted, expand_sweep

def test_set_dotted_creates_nested_value():
    d = {"optim": {"lr": 1e-3}}
    set_dotted(d, "optim.lr", 5e-4)
    assert d["optim"]["lr"] == 5e-4

def test_load_config_merges_override(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("optim:\n  lr: 0.001\n  epochs: 30\n")
    exp = tmp_path / "exp.yaml"
    exp.write_text("base: base.yaml\noverride:\n  optim:\n    lr: 0.0005\n")
    cfg = load_config(str(exp))
    assert cfg["optim"]["lr"] == 0.0005   # override applicato
    assert cfg["optim"]["epochs"] == 30   # campo non toccato preservato

def test_expand_sweep_cartesian_product():
    cfg = {"experiment": {"name": "p"}, "optim": {"lr": 0.001},
           "sweep": {"optim.lr": [0.001, 0.0005], "optim.epochs": [10, 20]}}
    runs = expand_sweep(cfg)
    assert len(runs) == 4                                  # 2 x 2
    assert all("sweep" not in r for r in runs)             # sweep rimosso dalle run concrete
    lrs = sorted(r["optim"]["lr"] for r in runs)
    assert lrs == [0.0005, 0.0005, 0.001, 0.001]

def test_expand_sweep_derives_unique_names():
    cfg = {"experiment": {"name": "base"}, "optim": {"lr": 0.001},
           "sweep": {"optim.lr": [0.001, 0.0005]}}
    names = {r["experiment"]["name"] for r in expand_sweep(cfg)}
    assert len(names) == 2                                 # nomi distinti
    assert all(n.startswith("base__") for n in names)

def test_expand_sweep_without_sweep_returns_single():
    cfg = {"experiment": {"name": "solo"}, "optim": {"lr": 0.001}}
    runs = expand_sweep(cfg)
    assert len(runs) == 1 and runs[0]["experiment"]["name"] == "solo"
