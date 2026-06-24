# Come funzionano Intent Classification e Slot Filling (Parte 2)

Doc concettuale (complementare a `docs/2026-06-24-part2a-nlu.md`, che spiega i file).
Qui rispondiamo a: **cosa produciamo? come? con quali dati? come c'entra l'attenzione?**

---

## 1. Il problema: capire una richiesta a un assistente voli

Un utente dice: *"voglio un volo da Boston a Denver domani mattina"*. Il sistema deve
capire **due cose diverse** sulla stessa frase:

1. **Intent** — *cosa vuole* l'utente, in generale: una **sola etichetta per l'intera
   frase** (es. `flight`). È un problema di **classificazione di testo**.
2. **Slot** — *i dettagli* da estrarre, parola per parola: una **etichetta per ogni
   token** (es. "Boston" → città di partenza, "Denver" → città di arrivo). È un
   problema di **sequence labeling** (etichettatura di sequenza).

Le impariamo **insieme** (multi-task) con un unico modello: condividono la stessa
comprensione della frase, quindi conviene un solo "cervello" con due "uscite".

---

## 2. Cosa produciamo, esattamente

Per la frase di esempio, l'output del modello è:

```
Input (token):   on    april                    first                     i  want  a  flight
Slot (per token):O     B-depart_date.month_name  B-depart_date.day_number  O  O     O  O
Intent (frase):  flight
```

- **Slot in formato IOB**: ogni token riceve `O` (nessuno slot), `B-x` (Begin: inizio
  di uno slot di tipo *x*) o `I-x` (Inside: continuazione di uno slot *x*). Esempio
  multi-parola: *"new york"* → `B-toloc.city_name I-toloc.city_name`. Questo schema
  permette di **segmentare** entità di più parole e insieme **etichettarle**.
- **Intent**: una sola etichetta scelta tra le 26 possibili (`flight`, `airfare`,
  `flight_time`, …).

Quindi il modello produce, per ogni frase: **una sequenza di label di slot** (lunga
quanto la frase) **+ una label di intent**.

---

## 3. I dati di training: ATIS

ATIS (Airline Travel Information Systems) = trascrizioni di persone che chiedono info
sui voli. Ogni esempio è un oggetto con tre campi:

```json
{
  "utterance": "what is the arrival time in san francisco for the 755 am flight leaving washington",
  "slots":     "O O O B-flight_time I-flight_time O B-fromloc.city_name I-fromloc.city_name O O B-depart_time.time I-depart_time.time O O B-fromloc.city_name",
  "intent":    "flight_time"
}
```

Numeri reali del nostro split:
- **4978** frasi di train, **893** di test;
- **26** classi di intent;
- **129** label di slot (IOB, inclusa `O`);
- frasi corte: **~11 token** in media (max 46).

**Dev set**: ATIS non ne fornisce uno; lo creiamo dal train (10%, **stratificato
sull'intent** → le proporzioni delle classi si conservano). Selezioniamo gli
iperparametri sul dev e teniamo il **test sigillato** per il numero finale.

**Da testo a numeri**: le reti lavorano con numeri, quindi mappiamo parole → id
(`word2id`), slot → id (`slot2id`), intent → id (`intent2id`) — è il ruolo della
classe `Lang`. Le parole vengono dal solo train (più un token `unk` per le parole mai
viste a test time); le label da tutto il corpus (così nessuna classe è "sconosciuta").

> Nota: in 2.A usiamo una **tokenizzazione a parole** (una parola = un token), quindi
> c'è corrispondenza 1:1 tra token e label di slot. In 2.B, con tokenizer a sub-parole
> (BPE), una parola può spezzarsi in più pezzi e nasce il problema di **allineare** le
> label ai sub-token — ma è un discorso del prossimo ciclo.

---

## 4. Come il modello produce i due output

Usiamo il **GPT2 della Parte 1** come "spina dorsale" (backbone): trasforma la frase in
una sequenza di **vettori contestuali**, uno per token. Sopra, montiamo **due teste**:

```
token ids ─► [embedding] ─► [N blocchi Transformer] ─► [LayerNorm] ─► h0 h1 ... hL   (un vettore per token)
                                                                        │        │
                                                            ┌───────────┘        └────────────┐
                                                            ▼                                 ▼
                                                   slot_out (per ogni h_t)          intent_out (solo su h_CLS)
                                                   ► label di slot per token        ► label di intent della frase
```

- **Testa slot** (`slot_out`, un `Linear`): applicata a **ogni** vettore `h_t` →
  produce, per ogni token, un punteggio su ciascuna delle 129 label di slot. L'argmax è
  lo slot predetto per quel token.
- **Testa intent** (`intent_out`, un `Linear`): applicata **solo** al vettore di un
  token speciale **CLS** → produce un punteggio su ciascuno dei 26 intent. L'argmax è
  l'intent della frase.

### Il token CLS (e perché sta in coda)
Appendiamo a ogni frase un token speciale `cls`. Il suo vettore finale serve a
rappresentare **l'intera frase** per la classificazione dell'intent. Lo mettiamo **in
coda** (ultima posizione) per un motivo legato all'attenzione causale — vedi sotto.

---

## 5. Come viene usata l'attenzione

L'attenzione è ciò che rende ogni vettore `h_t` **contestuale**: non rappresenta la
parola isolata, ma la parola **alla luce delle altre** rilevanti.

### Self-attention in una riga
Per ogni token si calcolano tre proiezioni: **query** (cosa cerco), **key** (cosa
offro), **value** (l'informazione che porto). Il vettore di un token diventa una
**media pesata dei value** degli altri token; i pesi vengono dalla somiglianza
query·key (normalizzata con softmax). In pratica: *"quanto ogni altra parola è
rilevante per me"*, e ne assorbo l'informazione.

È così che, ad esempio, il vettore di **"boston"** può incorporare il fatto che prima
c'era **"from"** → la testa slot capisce che è `B-fromloc.city_name` e non
`B-toloc.city_name`. Senza attenzione, "boston" sarebbe solo "una città", senza ruolo.

### Maschera causale (ereditata dal GPT2 della Parte 1)
Il nostro backbone è **decoder-only** come GPT2: l'attenzione è **mascherata**, cioè il
token in posizione *i* può guardare **solo** i token 0..*i* (il passato), mai il futuro.
Conseguenze:
- per gli **slot**: ogni token decide la sua label vedendo solo ciò che lo precede.
  Su ATIS funziona bene perché il contesto che disambigua di solito **precede** (es.
  "from boston", "to denver"). È però un limite: un contesto a destra non è visibile.
  La **Parte 2.B con BERT** (encoder *bidirezionale*) supera questo limite — un motivo
  per cui 2.B confronta decoder-only vs encoder-only.
- per l'**intent**: il token **CLS in coda** è l'ultimo, quindi con la maschera causale
  è **l'unico che vede tutta la frase**. Per questo lo mettiamo in fondo: il suo vettore
  finale è un riassunto dell'intero enunciato, perfetto per classificare l'intent.

### In sintesi
L'attenzione costruisce, per ogni token, una rappresentazione che **fonde la parola col
suo contesto**; la testa slot la usa token-per-token, mentre il CLS (che ha "visto"
tutto) alimenta la testa intent.

---

## 6. Come si allena (e si misura)

- **Loss multi-task** = `CrossEntropy(slot) + CrossEntropy(intent)` (pesi uguali). La
  loss degli slot **ignora** il padding e il token CLS (non sono slot veri).
- Non è generazione autoregressiva: è **etichettatura**. Diamo la frase intera, il
  modello produce tutte le label in un colpo; non serve "teacher forcing" come nel LM.
- **Metriche**: intent → **accuracy** (frazione di frasi con intent giusto); slot →
  **F1 a livello di chunk** con lo script `conll` (uno slot conta come corretto solo se
  l'intero "pezzo" multi-parola è esatto). Selezioniamo il modello sulla **slot F1 di
  dev** (il task più difficile) e riportiamo il numero finale sul **test**.

---

## 7. Riepilogo in tre frasi

1. **Cosa produciamo**: per ogni frase, una label di slot per token (IOB) + una label
   di intent per la frase.
2. **Come**: un GPT2 (backbone della Parte 1) trasforma la frase in vettori contestuali
   grazie all'**attenzione**; una testa li mappa in slot (per token) e una seconda testa
   mappa il vettore del **CLS** (in coda, che vede tutta la frase) nell'intent.
3. **Con quali dati**: ATIS (4978 train / 893 test, 26 intent, 129 slot IOB), con dev
   stratificato al 10% e test sigillato.
