# Lanciare le run della Parte 1.B (LoRA) sulla VM (tmux + workflow)

Guida pratica per far girare la pipeline **1.B** (GPT2 pre-addestrato + LoRA) sulla
VM GPU senza perdere il lavoro se cade la connessione SSH. È il gemello di
`lm_pipeline/RUNNING.md` (Parte 1.A): qui si usa `run_b.py`, le config in
`configs/experiments_b/`, e gli artefatti finiscono in `runs_b/` (separati da
`runs/` di 1.A).

> Differenza con 1.A: in 1.B il modello è grande (GPT2-small, ~124M parametri) ma
> alleniamo solo gli adapter (~442k). Al **primo avvio** la pipeline scarica i pesi
> di GPT2 da HuggingFace (poi resta in cache `~/.cache/huggingface`). Tieni il
> `batch_size` basso (default 8) per non saturare la memoria GPU.

## Perché tmux

Se lanci un comando nella normale sessione SSH e la connessione cade (timeout,
chiudi il portatile, cambi rete), il processo **muore** e perdi la run. `tmux` crea
una sessione che **vive sul server** indipendente dalla tua connessione: ti
"stacchi" (detach) e il processo continua; più tardi ti "riattacchi" (attach).

Concetti chiave:
- **detach**: ti scolleghi dalla sessione, ma resta viva sul server.
- **attach**: ti ricolleghi a una sessione esistente.
- Il prefisso dei comandi tmux è `Ctrl-b`: lo premi, lo rilasci, poi premi il
  tasto successivo.

## Workflow per OGNI run

### 1. Connettiti e apri/riusa la sessione tmux

```bash
ssh disi@<IP_VM>

# Crea una nuova sessione chiamata "lora":
tmux new -s lora

# (Se la sessione esiste già perché l'avevi creata prima:)
tmux attach -t lora
```

### 2. Dentro tmux: vai nel progetto e attiva l'ambiente

```bash
cd ~/NaturalLanguageUnderstandingProject
source .venv/bin/activate     # oppure: conda activate nlu26
```

### 3. (Consigliato la prima volta) verifica che la GPU sia viva

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```
Deve stampare `CUDA: True | Tesla V100-PCIE-16GB`.
Se stampa `False`, NON lanciare: vedi `memory/vm-gpu-setup.md` (driver / versione torch).

### 4. Lancia la run

```bash
# Sweep B0 baseline: cerca il learning rate (3 run, rank/alpha fissi):
python run_b.py sweep --config configs/experiments_b/00_baseline.yaml

# Oppure una singola run (config senza blocco sweep, o smoke test):
python run_b.py run --config configs/experiments_b/_smoke.yaml
```

> `sweep` vs `run`: usa **sweep** se il config ha un blocco `sweep:` (più valori da
> confrontare). `run` lancia una sola run e **ignora** il blocco sweep.

### 5. Staccati lasciando la run in esecuzione

Premi `Ctrl-b` poi `d` (detach). Torni alla shell normale; la run continua. Ora puoi
anche chiudere l'SSH: il processo sopravvive.

### 6. Ricontrollare l'avanzamento

Riconnettiti via SSH e riattacca la sessione:
```bash
tmux attach -t lora
```
Vedi l'output del training in tempo reale (incluso `param_stats`: trainable vs
frozen). Per staccarti di nuovo: `Ctrl-b` `d`.

## L'approccio incrementale di 1.B: lr → rank → alpha

Il lab chiede di sperimentare con `rank` e `alpha`. Procedi un parametro alla volta,
portando di volta in volta il valore migliore nel config successivo:

```bash
# 1) baseline: trova il miglior learning rate
python run_b.py sweep --config configs/experiments_b/00_baseline.yaml
python run_b.py aggregate     # guarda partB_summary.md per il lr migliore

# 2) scrivi il miglior lr in 01_rank_sweep.yaml (campo optim.lr), poi:
python run_b.py sweep --config configs/experiments_b/01_rank_sweep.yaml
python run_b.py aggregate     # scegli il rank migliore

# 3) scrivi il miglior rank in 02_alpha_sweep.yaml (campo model.rank), poi:
python run_b.py sweep --config configs/experiments_b/02_alpha_sweep.yaml
python run_b.py aggregate
```

## Quando le run sono FINITE

Dentro la cartella del progetto, con l'ambiente attivo:

```bash
python run_b.py aggregate   # raccoglie le run di runs_b/ -> partB_summary.* in reports/
python run_b.py finalize    # ri-allena la migliore sul dataset intero e valuta su test
python run_b.py export       # genera LM/part_B e copia gli adapter in LM/part_B/bin/
```

> `finalize` salva gli adapter della run definitiva in `runs_b/<nome>__FINAL/adapters.pt`;
> `export` rigenera `LM/part_B/main.py` con gli iperparametri vincenti. **Verifica**:
> `python LM/part_B/main.py` deve ristampare la test-PPL.

Gli artefatti delle singole run stanno in `runs_b/`, i report aggregati in
`reports/` (file `partB_*`). Per scaricarli sul tuo portatile (dal TUO terminale
locale, non dalla VM):
```bash
scp -r disi@<IP_VM>:~/NaturalLanguageUnderstandingProject/reports ./reports
# e gli adapter della consegna:
scp disi@<IP_VM>:~/NaturalLanguageUnderstandingProject/LM/part_B/bin/adapters.pt ./LM/part_B/bin/
```

Quando hai finito con la sessione tmux e non ti serve più:
```bash
tmux kill-session -t lora
```

## Notifica sul telefono quando la run finisce (ntfy.sh)

`ntfy.sh` manda una notifica push sul telefono con un solo `curl`. Gratis, nessuna
registrazione. (Setup identico a quello descritto in `lm_pipeline/RUNNING.md`:
installa l'app ntfy, iscriviti a un topic segreto, es. `nlu-teo-9f3kx72`.)

**Modo A — concatenato (il più semplice):** la notifica parte appena il comando
termina. Lancialo dentro tmux:
```bash
python run_b.py sweep --config configs/experiments_b/01_rank_sweep.yaml; \
curl -H "Title: LoRA rank sweep" -H "Tags: white_check_mark" \
     -d "Finito!" ntfy.sh/nlu-teo-9f3kx72
```
(`;` notifica sempre; usa `&&` per notificare solo se la run è andata a buon fine.)

**Modo B — guardiano (se la run è GIÀ partita):** aspetta che il processo finisca e
poi notifica. Apri una nuova finestra tmux (`Ctrl-b` `c`) e:
```bash
PID=$(pgrep -f "[r]un_b.py sweep" | head -1); echo "PID = $PID"
( while kill -0 $PID 2>/dev/null; do sleep 30; done; \
  N=$(ls ~/NaturalLanguageUnderstandingProject/runs_b/*/metrics.json 2>/dev/null | wc -l); \
  curl -H "Title: Sweep LoRA" -H "Tags: white_check_mark" \
       -d "Finito. metrics.json totali in runs_b: $N" \
       ntfy.sh/nlu-teo-9f3kx72 ) &
```
Poi stacca con `Ctrl-b` `d`.

## Alternativa privata: bot Telegram

Setup identico a `lm_pipeline/RUNNING.md` (crea il bot con @BotFather, ricava il
`chat_id`, esporta `TG_TOKEN`/`TG_CHAT`). Ad ogni run:

**Modo A — concatenato:**
```bash
python run_b.py sweep --config configs/experiments_b/00_baseline.yaml; \
curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
     -d chat_id="$TG_CHAT" -d text="Sweep LoRA baseline finito ✅"
```

**Modo B — guardiano (run già partita):** nuova finestra tmux (`Ctrl-b` `c`):
```bash
PID=$(pgrep -f "[r]un_b.py sweep" | head -1); echo "PID = $PID"
( while kill -0 $PID 2>/dev/null; do sleep 30; done; \
  N=$(ls ~/NaturalLanguageUnderstandingProject/runs_b/*/metrics.json 2>/dev/null | wc -l); \
  curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
       -d chat_id="$TG_CHAT" \
       -d text="Sweep LoRA finito. metrics.json totali in runs_b: $N" ) &
```
Poi stacca con `Ctrl-b` `d`.

> Nota: se usi le variabili `$TG_TOKEN`/`$TG_CHAT` nel guardiano, assicurati che
> siano esportate nella shell da cui lo lanci (o messe in `~/.bashrc`), altrimenti
> dentro il subshell risulteranno vuote.

## Comandi tmux di riferimento

| Azione                         | Comando                       |
|--------------------------------|-------------------------------|
| Nuova sessione "lora"          | `tmux new -s lora`            |
| Staccarsi (lasciando girare)   | `Ctrl-b` poi `d`              |
| Riattaccarsi                   | `tmux attach -t lora`         |
| Lista sessioni attive          | `tmux ls`                     |
| Chiudere una sessione          | `tmux kill-session -t lora`   |
| Scorrere l'output (scroll)     | `Ctrl-b` poi `[`, esci con `q`|

## Alternativa senza tmux (nohup)

Se non vuoi tmux, puoi lanciare in background con log su file:
```bash
nohup python run_b.py sweep --config configs/experiments_b/00_baseline.yaml > sweep_b.log 2>&1 &
tail -f sweep_b.log   # segui l'avanzamento; esci con Ctrl-c (non ferma la run)
```
tmux resta comunque più comodo perché ti fa rivedere il terminale interattivo.
