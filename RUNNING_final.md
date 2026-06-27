# Guida operativa — re-run pulito del progetto (branch `final`)

Tutto il lavoro "corretto" sta sul branch **`final`**. Questa guida ha i comandi
ordinati parte-per-parte. Regola d'oro:
- **Parte 2 (NLU)**: ricerca a seed singolo → **multi-seed (×5)** sui vincitori/finali
  (lo chiede il lab 5) → media ± std con `seed_summary.py`.
- **Parte 1 (LM)**: **seed singolo** (il lab 4 non chiede il multi-seed).
- Approccio incrementale "una cosa alla volta": a ogni step porti il valore vincente nel
  config successivo (i campi `<- ...` nei commenti indicano dove).

---

## 0a. Bootstrap nuova VM (Ubuntu 20.04, V100, senza conda — una tantum)

Driver NVIDIA gia' presente (verifica con `nvidia-smi`: deve mostrare la Tesla V100).
Il sistema ha Python 3.8; il nostro env e' 3.10.13 → si usa Miniconda nella home (no sudo).
**Auth GitHub via SSH** (la password HTTPS non e' piu' supportata):

```bash
# --- chiave SSH per GitHub (una volta) ---
ssh-keygen -t ed25519 -C "teomece@gmail.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub      # incolla su GitHub: Settings -> SSH and GPG keys -> New SSH key
ssh -T git@github.com          # atteso: "Hi TeoMece! You've successfully authenticated"

# --- Miniconda nella home (no sudo) ---
cd ~ && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate && conda init bash && source ~/.bashrc

# --- repo + ambiente ---
git config --global user.name "TeoMece" && git config --global user.email "teomece@gmail.com"
git clone git@github.com:TeoMece/NaturalLanguageUnderstandingProject.git
cd NaturalLanguageUnderstandingProject && git checkout final && git pull
conda create -y -n nlu26 python=3.10.13 && conda activate nlu26
pip install --upgrade pip && pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
python -m pytest -q             # 60 test verdi = ambiente sano
```

## 0b. Setup ad ogni sessione

```bash
ssh disi@<IP_VM>
tmux new -s final
cd ~/NaturalLanguageUnderstandingProject
conda activate nlu26
git fetch --all --prune && git checkout final && git pull
rm -rf runs runs_b runs_nlu runs_nlu_b        # clean slate delle run (solo al primo giro!)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # True
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
# ricerca lr, per modello: OGNI sweep gira 5 seed/valore (gia' nei config).
python run_nlu_b.py sweep --config configs/nlu_b/experiments/gpt2.yaml   # 15 run (3 lr x5)
python run_nlu_b.py sweep --config configs/nlu_b/experiments/bert.yaml   # 15 run; scarica bert al 1° uso
python seed_summary.py --runs runs_nlu_b      # miglior lr per modello sulla MEDIA
python run_nlu_b.py aggregate                 # tabella di dettaglio

# FINALE encoder vs decoder: crea 2 config (gpt2 best lr) e (bert best lr) + mode:final
#   (il seed sweep e' gia' nei config) -> seed_summary = test slot F1 / intent acc MEDIA ± std.
python run_nlu_b.py sweep --config configs/nlu_b/experiments/<gpt2_final>.yaml
python run_nlu_b.py sweep --config configs/nlu_b/experiments/<bert_final>.yaml
python seed_summary.py --runs runs_nlu_b --out reports/seed_summary_nlu_b.md
python run_nlu_b.py export           # NLU/part_B (pesi ~500MB -> zip, non git)
```

## 2) Parte 2.A — GPT2 da zero (`run_nlu.py`, `runs_nlu/`)

```bash
# Ricerca incrementale: OGNI sweep gira 5 seed/valore (gia' nei config). Dopo ogni step:
#   seed_summary.py -> scegli il vincitore sulla MEDIA della dev slot F1, poi propaga.
python run_nlu.py sweep --config configs/nlu/experiments/00_baseline.yaml   # 15 run (3 lr x5)
python seed_summary.py --runs runs_nlu          # scegli miglior lr sulla MEDIA
#  porta il miglior lr in 01_arch.yaml, poi:
python run_nlu.py sweep --config configs/nlu/experiments/01_arch.yaml       # 30 run (6 x5)
python seed_summary.py --runs runs_nlu
#  porta arch in 01b_ffn / 01c_heads:
python run_nlu.py sweep --config configs/nlu/experiments/01b_ffn.yaml ; python seed_summary.py --runs runs_nlu
python run_nlu.py sweep --config configs/nlu/experiments/01c_heads.yaml ; python seed_summary.py --runs runs_nlu
python run_nlu.py sweep --config configs/nlu/experiments/02_dropout.yaml ; python seed_summary.py --runs runs_nlu
python run_nlu.py aggregate          # tabella di dettaglio (tutte le run)

# FINALE: crea un config con gli iperparametri vincenti + sweep experiment.seed:[42,1,2,3,4]
#   + mode:final -> 5 run su test -> seed_summary = test slot F1 / intent acc MEDIA ± std.
python run_nlu.py sweep --config configs/nlu/experiments/03_final.yaml      # (da creare coi vincitori)
python seed_summary.py --runs runs_nlu --out reports/seed_summary_nlu.md
python run_nlu.py export             # NLU/part_A
```

## 3) Parte 1.A — GPT2 da zero (LM) (`run.py`, `runs/`) — SEED SINGOLO

```bash
# ricerca incrementale (lo Step 0 del lab E' la baseline con ricerca lr):
# lr -> [ablazione scheduler] -> arch -> ff -> heads -> dropout -> weight tying
python run.py sweep --config configs/experiments/00_baseline.yaml ; python run.py aggregate
# ablazione ricetta: scheduler OFF sui 3 lr (gemello del baseline) -> contributo dello
# scheduler e dipendenza dall'lr. Deciso qui una volta; resta ON per tutta la catena.
python run.py sweep --config configs/experiments/00b_no_scheduler.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/01_arch_dyn_ff.yaml ; python run.py aggregate
#  porta arch in 01b_ff.yaml e 01c_heads.yaml:
python run.py sweep --config configs/experiments/01b_ff.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/01c_heads.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/02_dropout.yaml ; python run.py aggregate
python run.py sweep --config configs/experiments/03_weight_tying.yaml ; python run.py aggregate
# (opzionali: 05_bigarch / 06_bigarch_dropout stress test Leva 1)
#  l'ablazione dello scheduler ora e' 00b_no_scheduler, fatta subito dopo il baseline.

python run.py finalize               # test del migliore
python run.py export                 # LM/part_A
```

## 4) Parte 1.B — LoRA (`run_b.py`, `runs_b/`) — SEED SINGOLO

```bash
# Step 0 - ZERO-SHOT: GPT2 pre-addestrato senza adapter (epochs=0) = riferimento "prima di LoRA"
python run_b.py run --config configs/experiments_b/00_zeroshot.yaml
# ricerca incrementale: lr -> rank -> alpha
python run_b.py sweep --config configs/experiments_b/00_baseline.yaml ; python run_b.py aggregate   # lr
#  porta il miglior lr in 01_rank_sweep.yaml:
python run_b.py sweep --config configs/experiments_b/01_rank_sweep.yaml ; python run_b.py aggregate # rank
#  porta rank in 02_alpha_sweep.yaml:
python run_b.py sweep --config configs/experiments_b/02_alpha_sweep.yaml ; python run_b.py aggregate # alpha
python run_b.py finalize             # test del migliore
python run_b.py export               # LM/part_B
```
> **Nota**: i placeholder `lr 5e-4 / rank 16` in `01_rank_sweep`/`02_alpha_sweep` sono
> residui del vecchio run: vanno riaggiornati coi vincitori del NUOVO sweep (lr → rank → alpha).
> Lo zero-shot va nel report come Step 0 ("quanto aggiunge LoRA"), NON nella catena di tuning.

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
