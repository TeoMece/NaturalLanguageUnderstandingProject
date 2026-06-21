# Lanciare le run sulla VM (tmux + workflow)

Guida pratica per far girare la pipeline sulla VM GPU senza perdere il lavoro
se cade la connessione SSH.

## Perché tmux

Se lanci un comando dentro la normale sessione SSH e la connessione cade
(timeout, chiudi il portatile, cambia rete), il processo **muore** e perdi la
run. `tmux` crea una sessione di terminale che **vive sul server** indipendente
dalla tua connessione: puoi "staccarti" (detach) e il processo continua a
girare; più tardi ti "riattacchi" (attach) e ritrovi tutto com'era.

Concetti chiave:
- **detach**: ti scolleghi dalla sessione, ma resta viva sul server.
- **attach**: ti ricolleghi a una sessione esistente.
- Il prefisso dei comandi tmux è `Ctrl-b`: lo premi, lo rilasci, poi premi
  il tasto successivo.

## Workflow per OGNI run

### 1. Connettiti e apri/riusa la sessione tmux

```bash
ssh disi@<IP_VM>

# Crea una nuova sessione chiamata "baseline":
tmux new -s baseline

# (Se la sessione esiste già perché l'avevi creata prima:)
tmux attach -t baseline
```

### 2. Dentro tmux: vai nel progetto e attiva il venv

```bash
cd ~/NaturalLanguageUnderstandingProject
source .venv/bin/activate
```

### 3. (Consigliato la prima volta) verifica che la GPU sia viva

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```
Deve stampare `CUDA: True | Tesla V100-PCIE-16GB`.
Se stampa `False`, NON lanciare: vedi `memory/vm-gpu-setup.md` (driver / versione torch).

### 4. Lancia la run

```bash
# Sweep (esperimento 0 baseline: 3 run, una per learning rate):
python run.py sweep --config configs/experiments/00_baseline.yaml

# Oppure una singola run (config senza blocco sweep, o smoke test):
python run.py run --config configs/experiments/00_baseline.yaml
```

> `sweep` vs `run`: usa **sweep** se il config ha un blocco `sweep:` (più valori
> da confrontare). `run` lancia una sola run e **ignora** il blocco sweep.

### 5. Staccati lasciando la run in esecuzione

Premi `Ctrl-b` poi `d` (detach). Torni alla shell normale; la run continua.
Ora puoi anche chiudere l'SSH: il processo sopravvive.

### 6. Ricontrollare l'avanzamento

Riconnettiti via SSH e riattacca la sessione:
```bash
tmux attach -t baseline
```
Vedi l'output del training in tempo reale. Per staccarti di nuovo: `Ctrl-b` `d`.

## Quando la run è FINITA

Dentro la cartella del progetto, col venv attivo:

```bash
python run.py aggregate   # raccoglie tutte le run -> tabelle/figure in reports/
python run.py finalize    # seleziona la run migliore e valuta su test (stampa PPL finale)
```

Gli artefatti delle singole run stanno in `runs/`, i report aggregati in
`reports/`. Per scaricarli sul tuo portatile (dal TUO terminale locale, non
dalla VM):
```bash
scp -r disi@<IP_VM>:~/NaturalLanguageUnderstandingProject/reports ./reports
```

Quando hai finito con la sessione tmux e non ti serve più:
```bash
tmux kill-session -t baseline
```

## Notifica sul telefono quando la run finisce (ntfy.sh)

`ntfy.sh` manda una notifica push sul telefono con un solo `curl`. Gratis,
nessuna registrazione.

### Setup (solo la PRIMA volta)

1. Installa l'app **ntfy** sul telefono (Android / iOS).
2. Nell'app: **Subscribe to topic** e scegli un nome **a caso e difficile da
   indovinare** (i topic sono pubblici: chi lo conosce può leggerlo).
   Esempio: `nlu-teo-9f3kx72`. Annotalo, è il "tuo canale".
3. Verifica che funzioni mandando un test dalla VM:
   ```bash
   curl -d "test notifica" ntfy.sh/nlu-teo-9f3kx72
   ```
   Se arriva sul telefono, sei pronto. (Sostituisci sempre il topic col tuo.)

### Ad OGNI run — due modi

**Modo A — concatenato (il più semplice):** la notifica parte appena il
comando termina. Lancialo dentro tmux:
```bash
python run.py sweep --config configs/experiments/02_dropout.yaml; \
curl -H "Title: Sweep dropout" -H "Tags: white_check_mark" \
     -d "Finito!" ntfy.sh/nlu-teo-9f3kx72
```
(`;` notifica sempre; usa `&&` per notificare solo se la run è andata a buon fine.)

**Modo B — guardiano (se la run è GIÀ partita):** aspetta che il processo
finisca e poi notifica. Apri una nuova finestra tmux (`Ctrl-b` `c`) e:
```bash
PID=$(pgrep -f "[r]un.py sweep" | head -1); echo "PID = $PID"
( while kill -0 $PID 2>/dev/null; do sleep 30; done; \
  N=$(ls ~/NaturalLanguageUnderstandingProject/runs/*/metrics.json 2>/dev/null | wc -l); \
  curl -H "Title: Sweep baseline" -H "Tags: white_check_mark" \
       -d "Finito su labEIOX5N. metrics.json totali: $N" \
       ntfy.sh/nlu-teo-9f3kx72 ) &
```
Poi stacca con `Ctrl-b` `d`.

## Alternativa privata: bot Telegram

A differenza di ntfy (topic pubblico, chiunque lo indovini legge), Telegram è
**privato**: solo tu ricevi i messaggi del tuo bot. Richiede un po' di setup in
più, ma una volta fatto è solido e comodo.

### Setup (solo la PRIMA volta)

1. **Crea il bot.** Sul telefono apri Telegram, cerca **@BotFather**, premi
   Start e manda `/newbot`. Segui le istruzioni (nome + username che finisce per
   `bot`). BotFather ti restituisce il **token**, una stringa tipo
   `123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`. **Tienilo segreto** (chi ha
   il token controlla il bot).
2. **Avvia una chat col bot.** Cerca il tuo bot per username e premi **Start** (o
   mandagli un messaggio qualsiasi). Senza questo passo il bot non può scriverti.
3. **Ricava il tuo `chat_id`.** Dalla VM (sostituisci `<TOKEN>`):
   ```bash
   curl -s "https://api.telegram.org/bot<TOKEN>/getUpdates" | grep -o '"chat":{"id":[0-9-]*'
   ```
   Cerca il numero dopo `"id":` (es. `123456789`). È il tuo **chat_id**.
   > Se l'output è vuoto: manda prima un messaggio al bot (passo 2), poi riprova.
4. **Salva token e chat_id come variabili** così non li riscrivi ogni volta
   (mettili anche in `~/.bashrc` per renderli permanenti):
   ```bash
   export TG_TOKEN="123456789:AAExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
   export TG_CHAT="123456789"
   ```
5. **Verifica** con un messaggio di prova:
   ```bash
   curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
        -d chat_id="$TG_CHAT" -d text="test notifica ✅"
   ```
   Se arriva su Telegram, sei pronto.

### Ad OGNI run

**Modo A — concatenato:**
```bash
python run.py sweep --config configs/experiments/00_baseline.yaml; \
curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
     -d chat_id="$TG_CHAT" -d text="Sweep baseline finito ✅ (labEIOX5N)"
```
(`;` notifica sempre; `&&` solo se la run è andata a buon fine.)

**Modo B — guardiano (run già partita):** nuova finestra tmux (`Ctrl-b` `c`):
```bash
PID=$(pgrep -f "[r]un.py sweep" | head -1); echo "PID = $PID"
( while kill -0 $PID 2>/dev/null; do sleep 30; done; \
  N=$(ls ~/NaturalLanguageUnderstandingProject/runs/*/metrics.json 2>/dev/null | wc -l); \
  curl -s "https://api.telegram.org/bot$TG_TOKEN/sendMessage" \
       -d chat_id="$TG_CHAT" \
       -d text="Sweep baseline finito su labEIOX5N. metrics.json totali: $N" ) &
```
Poi stacca con `Ctrl-b` `d`.

> Nota: se usi le variabili `$TG_TOKEN`/`$TG_CHAT` nel guardiano, assicurati che
> siano esportate nella shell da cui lo lanci (o messe in `~/.bashrc`), altrimenti
> dentro il subshell risulteranno vuote.

## Comandi tmux di riferimento

| Azione                         | Comando                       |
|--------------------------------|-------------------------------|
| Nuova sessione "baseline"      | `tmux new -s baseline`        |
| Staccarsi (lasciando girare)   | `Ctrl-b` poi `d`              |
| Riattaccarsi                   | `tmux attach -t baseline`     |
| Lista sessioni attive          | `tmux ls`                     |
| Chiudere una sessione          | `tmux kill-session -t baseline` |
| Scorrere l'output (scroll)     | `Ctrl-b` poi `[`, esci con `q`|

## Alternativa senza tmux (nohup)

Se non vuoi tmux, puoi lanciare in background con log su file:
```bash
nohup python run.py sweep --config configs/experiments/00_baseline.yaml > sweep.log 2>&1 &
tail -f sweep.log   # segui l'avanzamento; esci con Ctrl-c (non ferma la run)
```
tmux resta comunque più comodo perché ti fa rivedere il terminale interattivo.
