# Guida operativa — re-run pulito del progetto (branch `final`)

Tutto il lavoro "corretto" sta sul branch **`final`**. Questa guida ha i comandi
ordinati parte-per-parte. Regola d'oro:
- **Parte 2 (NLU)**: ricerca a seed singolo → **multi-seed (×5)** sui vincitori/finali
  (lo chiede il lab 5) → media ± std con `seed_summary.py`.
- **Parte 1 (LM)**: **seed singolo** (il lab 4 non chiede il multi-seed).
- Approccio incrementale "una cosa alla volta": a ogni step porti il valore vincente nel
  config successivo (i campi `<- ...` nei commenti indicano dove).

---

## 0. Setup sulla VM (una volta)

```bash
ssh disi@<IP_VM>
tmux new -s final
cd ~/NaturalLanguageUnderstandingProject
git fetch --all --prune && git checkout final && git pull
rm -rf runs runs_b runs_nlu runs_nlu_b        # clean slate delle run
source .venv/bin/activate                     # o: conda activate nlu26
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # True
pip install -r requirements.txt               # include scikit-learn
```

Ordine consigliato: **2.B → 2.A → 1.A → 1.B**. (2.B prima perché è l'unica mai girata.)

---

## Ricetta MULTI-SEED (solo Parte 2, per vincitori/finali)

Quando hai un **vincitore** (o il modello finale), ripetilo su 5 seed e fai la media:
1. crea un config con gli iperparametri vincenti + il blocco:
   ```yaml
   sweep:
     experiment.seed: [42, 1, 2, 3, 4]
   ```
   (per il numero di **test** finale aggiungi anche `mode: final` nell'override).
2. `python <run_script> sweep --config <quel_config>`  → 5 run `..._seed42`, ...
3. `python seed_summary.py --runs <runs_dir>`  → tabella **media ± std**.

---

## 1) Parte 2.B — fine-tuning GPT2 + BERT (`run_nlu_b.py`, `runs_nlu_b/`)

```bash
# ricerca lr, per modello (seed singolo)
python run_nlu_b.py sweep --config configs/nlu_b/experiments/gpt2.yaml
python run_nlu_b.py sweep --config configs/nlu_b/experiments/bert.yaml   # scarica bert al 1° uso
python run_nlu_b.py aggregate        # -> reports/partB_nlu_summary.md (scegli miglior lr per modello)

# MULTI-SEED del miglior gpt2 e del miglior bert (confronto encoder vs decoder con std)
#   crea 2 config = (gpt2 best lr) e (bert best lr) + sweep experiment.seed:[42,1,2,3,4] + mode: final
python run_nlu_b.py sweep --config configs/nlu_b/experiments/<gpt2_best_seeds>.yaml
python run_nlu_b.py sweep --config configs/nlu_b/experiments/<bert_best_seeds>.yaml
python seed_summary.py --runs runs_nlu_b --out reports/seed_summary_nlu_b.md

python run_nlu_b.py export           # NLU/part_B (pesi ~500MB -> zip, non git)
```

## 2) Parte 2.A — GPT2 da zero (`run_nlu.py`, `runs_nlu/`)

```bash
# ricerca incrementale (seed singolo): lr -> arch -> ff -> heads -> dropout
python run_nlu.py sweep --config configs/nlu/experiments/00_baseline.yaml ; python run_nlu.py aggregate
#  porta il miglior lr in 01_arch.yaml:
python run_nlu.py sweep --config configs/nlu/experiments/01_arch.yaml ; python run_nlu.py aggregate
#  porta arch in 01b_ffn.yaml:
python run_nlu.py sweep --config configs/nlu/experiments/01b_ffn.yaml ; python run_nlu.py aggregate
#  porta ff in 01c_heads.yaml:
python run_nlu.py sweep --config configs/nlu/experiments/01c_heads.yaml ; python run_nlu.py aggregate
#  porta heads in 02_dropout.yaml:
python run_nlu.py sweep --config configs/nlu/experiments/02_dropout.yaml ; python run_nlu.py aggregate

# MULTI-SEED del vincitore di ogni step + del finale (ricetta sopra) -> seed_summary
python seed_summary.py --runs runs_nlu --out reports/seed_summary_nlu.md

python run_nlu.py finalize           # test del migliore (meglio: multi-seed con mode:final)
python run_nlu.py export             # NLU/part_A
```

## 3) Parte 1.A — GPT2 da zero (LM) (`run.py`, `runs/`) — SEED SINGOLO

```bash
# ricerca incrementale (lo Step 0 del lab E' la baseline con ricerca lr):
# lr -> arch -> ff -> heads -> dropout -> weight tying
python run.py sweep --config configs/experiments/00_baseline.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/01_arch_dyn_ff.yaml ; python run.py aggregate
#  porta arch in 01b_ff.yaml e 01c_heads.yaml:
python run.py sweep --config configs/experiments/01b_ff.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/01c_heads.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/02_dropout.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/03_weight_tying.yaml ; python run.py aggregate
# (opzionali: 04_no_scheduler ablazione; 05_bigarch / 06_bigarch_dropout stress test Leva 1)

python run.py finalize               # test del migliore
python run.py export                 # LM/part_A
```

## 4) Parte 1.B — LoRA (`run_b.py`, `runs_b/`) — SEED SINGOLO

```bash
python run_b.py sweep --config configs/experiments_b/00_baseline.yaml ; python run_b.py aggregate   # lr
#  porta il miglior lr in 01_rank_sweep.yaml:
python run_b.py sweep --config configs/experiments_b/01_rank_sweep.yaml ; python run_b.py aggregate # rank
#  porta rank in 02_alpha_sweep.yaml:
python run_b.py sweep --config configs/experiments_b/02_alpha_sweep.yaml ; python run_b.py aggregate # alpha
python run_b.py finalize             # test del migliore
python run_b.py export               # LM/part_B
```
> **TODO 1.B — starting point zero-shot**: valutare GPT2 pre-addestrato **senza adapter**
> (es. una run con `epochs: 0`) per mostrare quanto aggiunge LoRA. Da implementare quando
> arriviamo a 1.B (piccola aggiunta al `fit`/eval di `lm_pipeline_b`).

---

## 5) Consegna (zip) e push

- I **report aggregati** (`reports/partA_summary.md`, `partB_nlu_summary.md`, ...,
  `seed_summary_*.md`) sono tracciati: `git add reports/ && git commit && git push` (su `final`).
- I **pesi** (`bin/*.pt`) sono gitignored. Per la consegna costruisci lo **zip sulla VM**
  (dove ci sono i .pt): `zip -r consegna_LM.zip LM` e `zip -r consegna_NLU.zip NLU`, poi
  `scp` sul portatile. (`.gitignore` non esclude i file dallo zip.)
- Sorgenti `LM/part_*`/`NLU/part_*` (`main.py`, `model.py`, ...) puoi committarli; i `.pt`
  no (troppo grandi per GitHub).

## 6) Report finali (1 pagina ciascuno, template INTERSPEECH)
- **LM** (1.A + 1.B): delta onesti (starting_point), figura gap overfitting; seed singolo.
- **NLU** (2.A + 2.B): numeri **media ± std** (multi-seed), confronto encoder vs decoder.
- Dichiarare l'uso di AI; compilare nome/matricola.

## Note
- `sweep` vs `run`: usa **sweep** se il config ha il blocco `sweep:`; `run` per una singola config.
- tmux: stacca `Ctrl-b d`, riattacca `tmux attach -t final`.
- Dopo ogni `aggregate`, committa il summary se vuoi che lo riveda e ti confermi le scelte.
