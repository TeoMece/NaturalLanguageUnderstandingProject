# Guida alla pipeline LM (Parte 1.A) — cosa fa ogni file e perché

Questa guida spiega, file per file, cosa stiamo costruendo e perché. Ogni volta che
incontriamo un "oggetto" tecnico (tokenizer, optimizer, scheduler, ...) ne diamo
anche una spiegazione breve. L'obiettivo è didattico: capire davvero cosa succede.

---

## 0. Il quadro generale: cosa stiamo facendo

Stiamo addestrando un **modello di linguaggio** (Language Model, LM): un modello che,
data una sequenza di parole, impara a **predire la parola successiva**. Se sa predire
bene la parola dopo, vuol dire che ha "capito" la struttura statistica della lingua.

Il modello che usiamo è un **GPT2** costruito da zero (decoder-only Transformer).
Lo addestriamo sul dataset **Penn Treebank (PTB)**, una raccolta di frasi in inglese.

La metrica con cui ci misuriamo è la **Perplexity (PPL)** — vedi sotto. Più è bassa,
meglio il modello predice. L'obiettivo del progetto è **PPL < 250** sul test set.

### Concetti chiave (glossario veloce)

- **Token**: l'unità minima di testo che il modello vede (non sempre una parola
  intera: spesso un pezzo di parola). Il testo viene spezzato in token dal *tokenizer*.
- **Tokenizer**: il componente che trasforma una stringa ("the cat") in una lista di
  numeri interi (gli ID dei token), e viceversa. Usiamo il tokenizer **BPE** di GPT2
  (Byte-Pair Encoding: impara a fondere coppie di caratteri frequenti in token unici).
- **Vocabolario**: l'insieme di tutti i token che il modello conosce. Quello di GPT2
  ha ~50.257 token. Il modello può predire solo token presenti nel vocabolario.
- **Embedding**: ogni token (un numero) viene trasformato in un **vettore** di numeri
  reali (es. 256 dimensioni). È così che il modello "ragiona" sui token: come vettori
  in uno spazio, dove token simili stanno vicini. Gli embedding sono *appresi*.
- **Perplexity (PPL)**: misura quanto il modello è "sorpreso" dal testo vero. Si
  calcola come `exp(loss media per token)`. Intuizione: una PPL di 100 significa che,
  in media, il modello è incerto come se dovesse scegliere tra 100 parole equiprobabili
  ad ogni passo. Più bassa = meno sorpresa = modello migliore.
- **Loss (funzione di perdita)**: un numero che misura *quanto il modello sbaglia*.
  L'addestramento consiste nel modificare i pesi del modello per ridurla. Per il LM
  usiamo la **Cross-Entropy**: penalizza il modello quando assegna bassa probabilità
  al token corretto.
- **Pesi / parametri**: i numeri interni del modello che vengono appresi durante il
  training (milioni di essi). Addestrare = trovare i valori dei pesi che minimizzano
  la loss.

---

## 1. La filosofia dell'architettura: due strati

Il codice è diviso in due strati, e questa divisione è il cuore del design:

- **Strato "puro"** (`data.py`, `model.py`, `train.py`): contiene solo il *cosa serve
  per addestrare e valutare il modello*. Non sa nulla di file di configurazione, di
  cartelle di output, di sweep. **Non importa mai** lo strato di orchestrazione.
- **Strato "orchestrazione"** (`config.py`, `tracking.py`, `experiment.py`,
  `aggregate.py`): coordina gli esperimenti — legge i config, crea le cartelle,
  salva i risultati, confronta le run. Sta *sopra* e usa lo strato puro.

**Perché questa divisione?** La consegna del progetto richiede file standalone
(`model.py`, `utils.py`, `functions.py`, `main.py`) che girino da soli, senza la nostra
infrastruttura. Siccome lo strato puro non dipende dall'orchestrazione, possiamo
**copiarlo tale e quale** nella struttura di consegna (è quello che fa l'export).
Una sola codebase, zero duplicazioni.

```
NLU_project_development/
  lm_pipeline/        <- il package Python con tutta la logica
    data.py           [puro]          dati: PTB, subset, dataloader
    model.py          [puro]          architettura GPT2
    train.py          [puro]          addestramento e valutazione
    config.py         [orchestrazione] lettura config YAML + sweep
    tracking.py       [orchestrazione] seed, cartelle, salvataggi
    experiment.py     [orchestrazione] esegue una run completa
    aggregate.py      [orchestrazione] tabelle e grafici dai risultati
  run.py              CLI: i comandi che lanci da terminale
  lm_pipeline_export.py   genera la struttura di consegna
  configs/            i file YAML che descrivono gli esperimenti
  runs/               output: una cartella per ogni run
  reports/            tabelle LaTeX/Markdown e grafici per il report
  dataset/PennTreeBank/   i file di testo di PTB
  tests/              i test automatici (uno per modulo)
```

---

## 2. `lm_pipeline/data.py` — i dati [strato puro]

**Ruolo**: prendere i file di testo di Penn Treebank e trasformarli in qualcosa che il
modello può digerire: batch di tensori numerici. Gestisce anche il sottocampionamento
(subset) per fare esperimenti veloci.

### Cosa fa, funzione per funzione

- **`read_file(path, eos_token)`**: legge il file (una frase per riga) e aggiunge in
  coda a ogni frase il marcatore `<eos>` ("end of sentence"). Serve a segnalare al
  modello dove finisce una frase.

- **`subset_sentences(sentences, fraction, max_samples, seed)`**: seleziona solo una
  parte delle frasi, in modo **riproducibile** (stesso `seed` → stesso sottoinsieme).
  Lo fa **per frasi intere**, mai spezzandole: nel LM la predizione avviene *dentro*
  la sequenza, quindi tagliare a metà una frase rovinerebbe i dati. È la leva per
  iterare in fretta su Mac (es. `fraction: 0.1` = usa il 10% delle frasi).
  - *Oggetto: `random.Random(seed)`* — un generatore di numeri casuali con un seme
    fisso. "Casuale ma riproducibile": serve perché un esperimento dia sempre lo
    stesso subset, altrimenti non potremmo confrontare le run in modo equo.

- **`PennTreeBank(Dataset)`**: un `Dataset` di PyTorch, cioè un oggetto che espone le
  frasi una a una (`__len__` dice quante sono, `__getitem__` restituisce la i-esima).
  - *Oggetto: `Dataset`* — l'astrazione standard di PyTorch per "una collezione di
    esempi". Serve perché il `DataLoader` (sotto) sappia come pescare gli esempi.

- **`collate_fn(batch, tokenizer)`**: prende un gruppetto di frasi (un *batch*) e:
  1. le **tokenizza** (stringhe → tensori di ID) con padding;
  2. costruisce `input_ids` e `labels` con uno **shift a sinistra di una posizione**.
     Esempio: per "I go to Miami", input = "I go to", target = "go to Miami". Cioè il
     modello, visto "I", deve predire "go"; visto "I go", deve predire "to"; ecc.
  3. conta i token reali non-padding (per normalizzare la loss correttamente —
     contiamo i token di *label*, perché è su quelli che la loss viene mediata).
  - *Oggetto: padding* — quando le frasi in un batch hanno lunghezze diverse, si
    "riempiono" quelle corte con un token speciale di riempimento, così formano una
    matrice rettangolare (i tensori devono essere rettangolari). Quei token finti
    vengono poi *ignorati* nel calcolo della loss.
  - *Oggetto: batch* — un piccolo gruppo di esempi processati insieme (es. 32 frasi).
    Si lavora a batch per efficienza (la GPU processa molti esempi in parallelo) e
    perché aggiornare i pesi su una media di più esempi rende l'addestramento stabile.

- **`get_tokenizer()`**: scarica e restituisce il tokenizer BPE di GPT2. Imposta
  `pad_token = eos_token` perché GPT2 non ha un token di padding nativo.
  - *Oggetto: tokenizer* — vedi glossario. Qui usiamo solo la sua capacità di
    spezzare il testo in token; NON usiamo i pesi pre-addestrati di GPT2 (in 1.A il
    modello si addestra da zero; i pesi pre-addestrati arriveranno nella Parte 1.B).

- **`build_dataloaders(cfg_data, dataset_dir, tokenizer)`**: assembla i tre
  **DataLoader** (train / validation / test) applicando il subset solo al train.
  - *Oggetto: `DataLoader`* — l'oggetto PyTorch che, dato un `Dataset`, produce i
    batch uno dopo l'altro (mescolandoli se `shuffle=True`). È il "nastro
    trasportatore" che alimenta il modello durante il training.
  - *Train / validation / test* — tre porzioni separate dei dati. **Train**: su cui il
    modello impara. **Validation**: per scegliere gli iperparametri e decidere quando
    fermarsi (il modello non impara su questi). **Test**: usato solo alla fine, per la
    metrica onesta da riportare. Tenerli separati evita di "barare" guardando i dati
    di test durante lo sviluppo.

---

## 3. `lm_pipeline/model.py` — l'architettura GPT2 [strato puro]

**Ruolo**: definire la rete neurale. È un **Transformer decoder-only** (lo stesso tipo
di architettura di GPT). Spieghiamo i mattoni dal basso verso l'alto.

### Concetti di base

- **Rete neurale**: una funzione con milioni di parametri (pesi) che trasforma un input
  (i token) in un output (le probabilità del token successivo). I pesi si apprendono.
- **Transformer**: un'architettura che, invece di leggere il testo parola per parola in
  ordine, guarda l'intera sequenza in parallelo e usa l'**attention** per decidere
  quali parole precedenti sono rilevanti per predire la prossima.
- **Decoder-only / causale**: il modello, per predire la parola in posizione *i*, può
  guardare solo le parole in posizione ≤ *i* (il passato), mai il futuro. Altrimenti
  "barerebbe" vedendo la risposta. Questo si realizza con una **maschera**.

### I mattoni, classe per classe

- **`MultiHeadAttention`** — il cuore del Transformer.
  - *Self-attention*: per ogni token, il modello calcola "quanto guardare" ogni altro
    token della sequenza. Lo fa con tre proiezioni dei vettori: **query** (cosa sto
    cercando), **key** (cosa offro), **value** (l'informazione che porto). Il prodotto
    query·key dà i "punteggi di attenzione", normalizzati con una **softmax** in pesi
    che sommano a 1; questi pesano i *value*.
    - *Oggetto: softmax* — una funzione che trasforma un vettore di numeri qualsiasi in
      una distribuzione di probabilità (valori positivi che sommano a 1). Qui serve a
      trasformare i punteggi grezzi in "quanta attenzione" dare a ciascun token.
  - *Multi-head ("più teste")*: invece di un solo meccanismo di attention, se ne usano
    diversi in parallelo (le "teste"), ognuno libero di concentrarsi su relazioni
    diverse (es. una testa sulla sintassi, una sui riferimenti). Gli output si
    concatenano e si proiettano.
  - *Maschera causale*: prima della softmax, i punteggi verso il futuro vengono messi a
    `-inf`, così la softmax li azzera → il token non può guardare avanti.
  - *Dropout (punti 2 e 3)*: vedi sotto. Qui sui pesi di attention e dopo la proiezione.

- **`FeedForward`** — dopo l'attention, ogni token passa per una piccola rete
  (Linear → GELU → Linear) applicata indipendentemente a ciascuna posizione. Serve a
  "elaborare" l'informazione raccolta dall'attention.
  - *Oggetto: Linear (layer lineare)* — una moltiplicazione matrice + bias: `y = Wx+b`.
    È il mattone base delle reti neurali; `W` e `b` sono pesi appresi.
  - *Oggetto: GELU* — una funzione di **attivazione** non-lineare (simile a ReLU ma più
    morbida). Le attivazioni non-lineari sono ciò che permette alla rete di imparare
    funzioni complesse; senza, una pila di Linear collasserebbe in un solo Linear.
  - *Dropout (punto 4)*: dopo l'ultimo Linear.

- **`TransformerBlock`** — un "blocco" = LayerNorm → Attention (+ residuo) → LayerNorm →
  FeedForward (+ residuo). Il GPT2 impila tanti di questi blocchi.
  - *Oggetto: LayerNorm (normalizzazione)* — riscala i valori in modo che abbiano media
    0 e varianza 1. Stabilizza e velocizza l'addestramento, evitando che i numeri
    crescano o si riducano troppo passando per molti strati.
  - *Oggetto: connessione residua (residuo / skip connection)* — invece di `x = f(x)`,
    si fa `x = x + f(x)`. Permette al gradiente di "scorrere" facilmente attraverso reti
    profonde, evitando che l'informazione si perda. È una delle idee chiave che rende
    addestrabili reti molto profonde.

- **`GPT2`** — il modello completo:
  1. **Embedding dei token** + **embedding posizionali** (il Transformer di per sé non
     sa l'ordine delle parole, quindi gli aggiungiamo un vettore che codifica la
     posizione 0, 1, 2, ...).
  2. **Dropout (punto 1)** sugli embedding.
  3. Una pila di `num_layers` blocchi Transformer.
  4. Una LayerNorm finale e un **`lm_head`** (un Linear che proietta sul vocabolario):
     produce, per ogni posizione, un punteggio per ciascuno dei ~50k token possibili.
     Questi punteggi (i **logit**) diventano probabilità con una softmax.
  - **Weight tying** (flag opzionale, miglioria di 1.A): far condividere gli stessi
    pesi tra l'embedding dei token in ingresso (`token_embed`) e il layer di uscita
    (`lm_head`). Intuizione: "la rappresentazione di una parola in entrata e in uscita
    è la stessa cosa". Riduce i parametri e spesso migliora la PPL.

- **I 4 punti di dropout** (miglioria di 1.A):
  - *Oggetto: dropout* — durante il training, "spegne" casualmente una frazione di
    neuroni ad ogni passo. Sembra controintuitivo, ma costringe la rete a non dipendere
    troppo da singoli neuroni → **regolarizzazione**: riduce l'overfitting (vedi sotto)
    e migliora la generalizzazione. In valutazione il dropout è disattivato.
    I 4 punti richiesti: (1) post-embedding, (2) sui pesi di attention, (3) dopo la
    proiezione dell'attention, (4) dopo l'ultimo Linear del FeedForward.

- **`init_weights(mat)`** — inizializza i pesi dei Linear con valori casuali piccoli
  (uniformi in ±0.01). I pesi vanno inizializzati prima del training; il *come* influisce
  sulla velocità di convergenza (questa è l'init del lab; un'alternativa comune è la
  gaussiana N(0, 0.02) di GPT2).

> *Overfitting*: quando il modello "memorizza" i dati di training invece di
> generalizzare → va benissimo sul train ma male su validation/test. Dropout, weight
> tying ed early stopping sono tecniche per contrastarlo.

---

## 4. `lm_pipeline/train.py` — addestramento e valutazione [strato puro]

**Ruolo**: il ciclo che fa effettivamente imparare il modello, e quello che lo valuta.
Riceve oggetti già pronti (modello, dataloader, iperparametri) e non sa nulla di config.

### Come impara una rete neurale (in breve)

1. Si passa un batch nel modello → si ottengono i logit (forward pass).
2. Si calcola la **loss** (quanto sbaglia rispetto ai target).
3. **Backpropagation**: si calcola il **gradiente** della loss rispetto a ogni peso —
   cioè "in che direzione muovere ogni peso per ridurre la loss".
4. L'**optimizer** aggiorna i pesi nella direzione indicata dal gradiente.
5. Si ripete per molti batch e molte **epoche** (un'epoca = un passaggio completo su
   tutti i dati di training).

### Cosa fa, funzione per funzione

- **`pick_device(requested)`**: sceglie dove far girare i calcoli.
  - *Oggetto: device* — `cpu` (lento), `cuda` (GPU Nvidia, veloce), `mps` (GPU Apple
    Silicon). `auto` sceglie il migliore disponibile. I tensori e il modello devono
    stare sullo stesso device.

- **`warmup_cosine_lambda(warmup_steps, total_steps)`**: definisce come varia il
  **learning rate** nel tempo.
  - *Oggetto: learning rate (lr)* — il "passo" con cui l'optimizer aggiorna i pesi.
    Troppo grande → l'addestramento diverge; troppo piccolo → lentissimo. È
    l'iperparametro più importante da tarare.
  - *Oggetto: scheduler* — un componente che fa **variare il learning rate durante il
    training** invece di tenerlo fisso. Il nostro: **warmup + cosine decay**.
    - *Warmup*: nei primi step il lr sale gradualmente da ~0 al valore target. Evita
      instabilità iniziale quando i pesi sono casuali e i gradienti grandi.
    - *Cosine decay*: dopo il warmup, il lr scende seguendo una curva coseno fino a ~0.
      La discesa morbida aiuta a stabilizzarsi in un buon minimo.
  - *Oggetto: step* — un singolo aggiornamento dei pesi (un batch). Diverso dall'epoca,
    che sono tutti i batch del dataset.

- **`_build_optimizer(model, cfg_optim)`**: crea l'optimizer.
  - *Oggetto: optimizer* — l'algoritmo che decide *come* aggiornare i pesi dato il
    gradiente. Il default è **AdamW**: una variante di Adam che adatta il passo per
    ogni peso individualmente e gestisce bene il *weight decay* (una piccola spinta a
    tenere i pesi piccoli, altra forma di regolarizzazione). È robusto e molto usato.

- **`train_one_epoch(...)`**: un giro completo sui dati di training. Per ogni batch:
  azzera i gradienti, forward, calcola la loss, backpropagation, **gradient clipping**,
  aggiorna i pesi (optimizer), avanza lo scheduler.
  - *Oggetto: gradient clipping* — "taglia" i gradienti troppo grandi a una soglia
    massima. Evita gli "exploding gradients" (aggiornamenti enormi che fanno saltare
    l'addestramento), un problema comune nei Transformer.

- **`evaluate(loader, criterion, model, device)`**: valuta il modello su validation o
  test **senza aggiornare i pesi** (`torch.no_grad()` disattiva il calcolo dei
  gradienti, risparmiando memoria). Restituisce la **PPL** = `exp(loss media per token)`.
  - *Oggetto: criterion / CrossEntropyLoss* — la funzione di loss. Per il LM confronta i
    logit predetti col token vero e penalizza le predizioni sbagliate. `ignore_index`
    le dice di ignorare i token di padding.

- **`fit(model, train_loader, valid_loader, cfg_optim, device, pad_id)`**: orchestra
  l'intero addestramento su più epoche, con **early stopping**, e restituisce lo
  **storico** (loss e PPL per epoca) più i pesi migliori.
  - *Oggetto: early stopping* — ferma il training se la PPL di validazione non migliora
    per `patience` epoche di fila. Evita di sprecare tempo e di andare in overfitting:
    si tiene la versione del modello che era migliore sulla validation, non l'ultima.

---

## 5. `lm_pipeline/config.py` — le configurazioni [orchestrazione]

**Ruolo**: leggere i file YAML che descrivono un esperimento e trasformarli in
dizionari Python pronti all'uso. Gestisce ereditarietà e sweep.

- *Oggetto: YAML* — un formato di file testuale leggibile per descrivere dati
  strutturati (chiavi e valori annidati). Lo usiamo per separare i *parametri* di un
  esperimento dal *codice*: cambi un numero nel YAML e hai un nuovo esperimento, senza
  toccare Python.

- **`load_config(path)`**: carica un file YAML. Se contiene `base: ...`, eredita prima
  da quel file (i default) e poi applica la sezione `override` (un **deep-merge**:
  sovrascrive solo i campi indicati, lasciando intatti gli altri). Così ogni
  esperimento è breve: dichiara solo ciò che cambia rispetto al baseline.

- **`set_dotted(d, "a.b.c", v)`**: imposta un valore annidato usando una chiave "con i
  punti". Serve allo sweep per scrivere valori in profondità (es. `optim.lr`).

- **`expand_sweep(cfg)`**: se il config ha una sezione `sweep` (liste di valori per uno
  o più parametri), genera **tutte le combinazioni** (il prodotto cartesiano), una
  config per combinazione, con un nome derivato univoco.
  - *Oggetto: sweep / grid search* — provare sistematicamente molte combinazioni di
    iperparametri lanciando un training per ciascuna, per poi confrontarle. "Spazzata"
    dello spazio dei parametri. In 1.A lo usiamo in modo mirato (un parametro alla
    volta) per rispettare l'approccio incrementale richiesto.
  - *Oggetto: prodotto cartesiano* — tutte le combinazioni possibili: `lr=[a,b]` ×
    `layers=[4,6]` → 4 combinazioni.

---

## 6. `lm_pipeline/tracking.py` — riproducibilità e salvataggi [orchestrazione]

**Ruolo**: tutto ciò che riguarda la *documentazione* di una run e la *riproducibilità*.

- **`set_seed(seed)`**: fissa il "seme" dei generatori casuali (Python, NumPy, PyTorch).
  - *Oggetto: seed / riproducibilità* — l'addestramento usa numeri casuali (init dei
    pesi, shuffle dei dati, dropout). Fissando il seed, due esecuzioni danno lo stesso
    risultato: indispensabile per confrontare esperimenti in modo equo e per poter
    rifare una run.

- **`git_hash()`**: registra l'identificativo della versione del codice (se in un repo
  git). Serve a sapere *con quale codice* è stata prodotta una run.

- **`make_run_dir(runs_root, name)`**: crea una cartella dedicata per la run (con
  suffisso incrementale se il nome esiste già), così ogni esperimento ha i suoi
  artefatti separati.

- **`save_metrics(run_dir, metrics)`**: salva un `metrics.json` con tutto ciò che serve
  al report: PPL migliore, PPL per epoca, numero di parametri, tempo, device, seed,
  config completa, ecc.
  - *Oggetto: JSON* — un formato testuale per salvare dizionari/dati strutturati.
    Facile da rileggere sia a occhio sia da codice (è la base per `aggregate`).

- **`save_curves(run_dir, hist)`**: salva le curve di addestramento in `curves.csv`
  (i numeri) e `curves.pdf` (il grafico della PPL per epoca, pronto per il report).
  - *Oggetto: matplotlib (backend "Agg")* — la libreria con cui disegniamo i grafici.
    Il backend "Agg" disegna su file senza bisogno di una finestra (funziona anche su
    server/headless).

---

## 7. `lm_pipeline/experiment.py` — una run completa [orchestrazione]

**Ruolo**: è la "colla" che mette insieme tutto. Data una config, esegue una run
intera: dati → modello → training → valutazione → salvataggio.

- **`already_done(runs_root, name)`**: dice se una run è già stata completata (esiste il
  suo `metrics.json`). Serve alla **resumability**: se uno sweep si interrompe a metà
  (es. chiudi il Mac), rilanciandolo le run già fatte vengono saltate.

- **`run_experiment(cfg, runs_root, dataset_dir, tokenizer=None)`**: il flusso completo:
  1. se già fatta, salta;
  2. fissa il seed e sceglie il device;
  3. costruisce i dataloader (applicando il subset; in `mode: final` usa il dataset
     intero);
  4. costruisce il modello GPT2 con gli iperparametri della config e lo inizializza;
  5. lo addestra con `fit` (early stopping);
  6. se `mode: final`, valuta sul **test set** (la metrica onesta da riportare);
  7. salva curve, pesi migliori (`best_model.pt`) e `metrics.json`.
  - *`mode: dev` vs `final`* — in `dev` (sviluppo) si valuta sulla validation e si può
    usare il subset, per iterare in fretta. In `final` si usa tutto il dataset e si
    valuta sul test: è la run definitiva che produce il numero del report.

---

## 8. `lm_pipeline/aggregate.py` — risultati per il report [orchestrazione]

**Ruolo**: leggere tutte le run e produrre materiale pronto per il report.

- **`collect_runs(runs_root)`**: scansiona tutte le cartelle di run, legge i
  `metrics.json` e li mette in una lista **ordinata per PPL crescente** (la migliore
  in cima).

- **`to_markdown(rows)`** e **`to_latex(rows)`**: generano la stessa tabella in due
  formati. Il Markdown per consultarla al volo; il **LaTeX** (stile *booktabs*) da
  incollare direttamente nel report scritto col template IEEE.
  - *Oggetto: LaTeX* — il sistema di composizione tipografica con cui si scrive il
    report. Le tabelle vanno fornite in una sintassi specifica: `to_latex` te le
    genera già pronte.

- **`write_reports(runs_root, reports_dir)`**: scrive su disco `partA_summary.tex`,
  `partA_summary.md` e un grafico comparativo (`figures/comparison.pdf`) che confronta
  la PPL di tutte le run a colpo d'occhio.

---

## 9. `run.py` — l'interfaccia a riga di comando (CLI)

**Ruolo**: è il file che *lanci tu* dal terminale. Traduce i comandi che scrivi nelle
chiamate alle funzioni della pipeline.

- *Oggetto: CLI / argparse* — una *Command Line Interface* è il modo di usare un
  programma da terminale scrivendo comandi e opzioni. `argparse` è la libreria standard
  di Python che interpreta quello che scrivi (es. `sweep --config X.yaml`) e lo passa al
  codice. I "verbi" (`run`, `sweep`, ...) si chiamano **sottocomandi**.

- **`build_parser()`**: definisce i sottocomandi e le opzioni:
  - `run --config X` → esegue una singola config;
  - `sweep --config X` → espande lo sweep ed esegue tutte le combinazioni;
  - `aggregate` → genera tabelle e grafici;
  - `finalize [--run NOME]` → run definitiva (vedi sotto);
  - `export` → genera la struttura di consegna.

- **`select_best_run(runs_root)`**: trova la run con la PPL di validazione più bassa.

- **`_finalize(run_name)`**: prende la run migliore (automaticamente, o quella forzata
  con `--run`), la ri-addestra in `mode: final` sull'intero dataset, valuta sul test e
  stampa la PPL finale.

- **`main()`**: smista il comando ricevuto alla funzione giusta.

---

## 10. `lm_pipeline_export.py` — la struttura di consegna

**Ruolo**: generare i file standalone richiesti dalla consegna del progetto.

- **`export_part_a(...)`**: copia i tre moduli puri rinominandoli come vuole la
  consegna (`model.py`→`model.py`, `data.py`→`utils.py`, `train.py`→`functions.py`),
  **genera** un `main.py` che congela inline gli iperparametri della config migliore,
  carica il modello salvato e stampa la PPL. Copia anche il modello (`bin/`) e il
  dataset. Lo strato di orchestrazione non viene copiato: non serve alla consegna, e
  siccome i moduli puri non lo importano, i file generati girano da soli.
  - *Perché generare `main.py` invece di copiarlo?* Perché deve "ricordare" la
    configurazione vincente (es. quanti layer, che lr) senza dipendere dai nostri YAML:
    i valori vengono scritti direttamente nel codice.

---

## 11. `configs/` — gli esperimenti descritti in YAML

- **`base_partA.yaml`**: i valori di default di tutto (dati, modello, ottimizzazione).
  Tutti gli esperimenti ereditano da qui.
- **`experiments/00_baseline.yaml`** ... **`04_no_scheduler.yaml`**: l'approccio
  **incrementale** richiesto da 1.A — ogni file aggiunge *una* modifica rispetto al
  precedente: ricerca del lr, poi architettura, poi dropout, poi weight tying, infine
  un'ablazione (scheduler off) per documentarne il contributo.

---

## 12. `tests/` — i test automatici

**Ruolo**: verificare che ogni pezzo faccia ciò che deve, in automatico.

- *Oggetto: test / pytest* — piccole funzioni che controllano un comportamento atteso
  (es. "il subset col seed 7 è deterministico"). `pytest` le esegue tutte e segnala se
  qualcosa si rompe. Abbiamo seguito il **TDD** (Test-Driven Development): prima il
  test (che fallisce), poi il codice che lo fa passare. Garantisce che il codice sia
  verificabile e che future modifiche non rompano ciò che funziona.

C'è un file di test per modulo (`test_config.py`, `test_data.py`, ...). In totale 32
test, tutti verdi.

---

## 13. Il ciclo di lavoro tipico (mettendo tutto insieme)

```bash
# 1. attiva l'ambiente con le librerie giuste
conda activate nlu26

# 2. cerca un buon learning rate (3 run automatiche)
python run.py sweep --config configs/experiments/00_baseline.yaml

# 3. guarda i risultati: tabella LaTeX/Markdown + grafico in reports/
python run.py aggregate

# 4. ripeti per le migliorie successive (architettura, dropout, weight tying),
#    portando di volta in volta i parametri migliori nel config successivo

# 5. run definitiva: ri-allena la migliore sull'intero dataset e valuta sul test
python run.py finalize

# 6. genera la struttura di consegna
python run.py export
```

Per iterare in fretta su Mac, abbassa `data.fraction` (es. 0.1) nei config di
sviluppo; per i numeri veri rimettilo a 1.0 (o usa `finalize`, che lo fa da solo).

---

## 14. Riepilogo mentale

- **data.py** prepara i token. **model.py** è il cervello (GPT2). **train.py** lo fa
  imparare e lo misura (PPL). Questi tre sono "puri" e finiscono nella consegna.
- **config.py / tracking.py / experiment.py / aggregate.py** orchestrano gli
  esperimenti: leggono i parametri, lanciano le run, salvano e confrontano i risultati.
- **run.py** è il telecomando; **export** prepara la consegna.
- L'obiettivo: trovare, con esperimenti incrementali documentati, la configurazione che
  porta la **Perplexity sotto 250** sul test set.
