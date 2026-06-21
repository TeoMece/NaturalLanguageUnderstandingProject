# Leva 1 — Init stile GPT-2 con *residual scaling* (spiegata bene)

> Obiettivo di questo documento: capire **perché** i modelli profondi nel nostro
> setup si allenano peggio di quelli bassi, e **come** un dettaglio
> dell'inizializzazione (un fattore `1/√(2N)`) lo risolve. Niente fede: ogni
> affermazione è accompagnata dal *perché*, con schemi e numeri.

---

## 0. L'indizio da cui parte tutto

Nel nostro sweep sull'architettura è successo questo:

```
   num_layers:   2      4      6
   valid PPL:  MEGLIO  ...   PEGGIO
```

Un modello a **2 layer batte uno a 6**. In un Transformer è **anomalo**: più
profondità dovrebbe dare risultati *almeno uguali* (i layer in più possono sempre
"imparare a non fare nulla"). Quando invece la profondità **peggiora**, il
sospetto numero uno è: **i modelli profondi non si stanno allenando bene**. E la
causa più comune è un'**inizializzazione che non tiene conto della profondità**.

---

## 1. Il "residual stream": l'autostrada del Transformer

Ogni blocco Transformer non *sostituisce* la rappresentazione: ci **somma** sopra
un contributo. È la connessione residuale (`x = x + qualcosa`).

Schema di **un** blocco (semplificato, come nel nostro `model.py`):

```
   x ──────────────────────────────────────┐  (residual: passa "dritto")
   │                                        │
   ├─> LayerNorm ─> Self-Attention ─> out_proj ─(+)─> x'   ← 1° scrittura nel residuo
   │                                        │
   x' ─────────────────────────────────────┐
   │                                        │
   ├─> LayerNorm ─> FeedForward(L1→GELU→L2) ─(+)─> x''      ← 2° scrittura nel residuo
```

Punti chiave:
- La `x` viaggia su un'**autostrada** (il *residual stream*) che attraversa tutti
  i blocchi.
- Ogni blocco ci **immette** due contributi: uno dall'attention (`out_proj`) e uno
  dalla feed-forward (il secondo `Linear`). Questi due strati sono le **"bocche"
  che scrivono nel residuo**.
- Con `N` blocchi → **2N scritture** totali sull'autostrada.

Impilando i blocchi:

```
 x0 ─(+)─(+)─(+)─(+)─ ... ─(+)─(+)─> xL
      b1   b1   b2   b2        bN   bN
     attn  ff  attn  ff       attn  ff
     └──────────── 2N contributi sommati ────────────┘
```

---

## 2. Il problema: la varianza **cresce con la profondità**

Domanda chiave: se ogni contributo che immettiamo ha una certa "ampiezza"
(deviazione standard ≈ `s`), **quanto diventa grande la `x` dopo 2N somme?**

Regola della statistica: sommando quantità (circa) indipendenti, **le varianze si
sommano**. Quindi:

```
   Var(x dopo 2N contributi) ≈ 2N · s²
   std(x)                    ≈ √(2N) · s
```

La deviazione standard del residuo **cresce come √(2N)**: più layer = autostrada
più "rumorosa/larga". Vediamolo con numeri (poniamo `s = 1` per illustrare):

| num_layers (N) | scritture (2N) | std del residuo ≈ √(2N) |
|:---:|:---:|:---:|
| 2 | 4  | **2.00** |
| 4 | 8  | **2.83** |
| 6 | 12 | **3.46** |
| 8 | 16 | **4.00** |

Un modello a 6 layer parte con un residuo **~3.5× più "gonfio"** di uno a 2 layer
— e cresce **senza limite** con la profondità. *Ecco perché 6 layer va peggio di
2*: non è la profondità in sé, è che la profondità, **senza correzione**, sballa
la scala dei segnali fin dall'inizializzazione.

---

## 3. Perché una varianza che cresce fa male (intuizione)

Due effetti, entrambi al **momento dell'inizializzazione** (il punto di partenza
dell'addestramento):

1. **Si parte lontano dall'"identità".** Una rete profonda ben condizionata
   dovrebbe partire **quasi come una funzione identità**: il residuo passa quasi
   dritto e ogni blocco aggiunge un contributo *piccolo e controllato*. Se invece
   ogni blocco immette tanto, la rete profonda parte in un regime "caotico" da cui
   è difficile imparare.

2. **I blocchi profondi "contano" sempre meno.** Man mano che il residuo si gonfia
   (√(2N)), il contributo *relativo* di ogni nuovo blocco diventa una frazione
   sempre più piccola del totale. Risultato: i layer profondi faticano a
   influenzare l'output → **profondità sprecata** (il modello si comporta come se
   fosse più basso, o instabile).

> Nota: il LayerNorm a valle normalizza la scala, ma **non annulla** questi
> problemi di condizionamento durante il training; il danno è nella *dinamica* di
> apprendimento, non solo nella scala finale.

---

## 4. La cura: scalare le "bocche" del residuo di `1/√(2N)`

Vogliamo che lo std del residuo resti **costante**, indipendente dalla profondità.
Idea: **ridurre l'ampiezza di ogni contributo** dello stesso fattore con cui
cresce, cioè `1/√(2N)`.

Come si riduce l'ampiezza di un `Linear`? Lo std dell'output di un `Linear` è
proporzionale allo **std dei suoi pesi**. Quindi basta inizializzare i pesi delle
sole **proiezioni che scrivono nel residuo** con:

```
   std = 0.02 / √(2N)            (N = num_layers)
```

invece del normale `0.02`. Verifichiamo che funziona — rifacciamo il conto con il
contributo ridotto `s' = s / √(2N)`:

```
   std(x) ≈ √(2N) · s'  =  √(2N) · (s / √(2N))  =  s     ← COSTANTE!
```

La √(2N) della crescita e la `1/√(2N)` dello scaling **si annullano**. Con la
correzione la tabella diventa:

| num_layers (N) | std SENZA scaling | std CON scaling `1/√(2N)` |
|:---:|:---:|:---:|
| 2 | 2.00 | **1.00** |
| 4 | 2.83 | **1.00** |
| 6 | 3.46 | **1.00** |
| 8 | 4.00 | **1.00** |

Ora un modello a 8 layer parte con la **stessa scala di segnale** di uno a 2 →
la profondità può finalmente *aiutare* invece di *sabotare*. È esattamente il
trucco del paper di GPT-2: *"scaliamo i pesi dei layer residuali di 1/√N, dove N è
il numero di layer residuali"*.

---

## 5. Due dettagli sull'init "normale a 0.02"

Cambiamo anche la base, non solo lo scaling:

| | init attuale (lab) | init proposto (GPT-2) |
|---|---|---|
| forma | `uniform(-0.01, +0.01)` | `normal(0, 0.02)` |
| std | `0.02/√12 ≈` **0.0058** | **0.02** |
| dove | tutti i `Linear` | tutti i `Linear`... |
| eccezione | — | ...ma le 2 proiezioni residuali a `normal(0, 0.02/√(2N))` |

- **Perché `normal(0, 0.02)`**: è lo standard empirico per i Transformer a queste
  larghezze — abbastanza grande da far "girare" il segnale, abbastanza piccolo da
  non esplodere. L'`uniform(±0.01)` del lab è **molto più piccolo** (0.0058) e
  piatto: un punto di partenza più debole.
- **Importante**: tutto questo vale **solo all'inizializzazione**. Durante il
  training i pesi si muovono liberamente; lo scaling serve solo a partire da un
  punto **ben condizionato**.

---

## 6. Dove interviene nel NOSTRO codice

Le due "bocche" del residuo sono già lì, vanno solo inizializzate diversamente:

```
 model.py
 ├─ MultiHeadAttention
 │    self.out_proj = nn.Linear(d_model, d_model)      ← riga ~40  [residuale → scala 1/√(2N)]
 │
 └─ FeedForward  (Linear → GELU → Linear)
      nn.Linear(d_model, hidden_dim)                   ← riga ~129 [normale → 0.02]
      nn.Linear(hidden_dim, d_model)                   ← riga ~131 [residuale → scala 1/√(2N)]
```

Schema di quali pesi prendono quale init:

```
   q,k,v, w_q/w_k/w_v, FFN.Linear1, lm_head ......... normal(0, 0.02)
   out_proj (attn),    FFN.Linear2  ................. normal(0, 0.02 / √(2·num_layers))   ← residuali
   token_embed, pos_embed ........................... normal(0, 0.02)   (già fatto nel fix precedente)
   bias, LayerNorm .................................. 0 / standard
```

Pseudocodice del nuovo `init_weights` (tying-aware come ora):

```python
residual_std = 0.02 / sqrt(2 * num_layers)

for ogni modulo m:
    if m è un Linear "residuale" (out_proj o FFN.Linear2):
        normal_(m.weight, std=residual_std)      # ← lo scaling
    elif m è un Linear:
        normal_(m.weight, std=0.02)
    elif m è un Embedding:
        normal_(m.weight, std=0.02)
    # i Linear con peso condiviso (weight tying) restano saltati, come ora
```

> Dettaglio implementativo: per riconoscere i Linear "residuali" serve marcarli
> (es. un attributo `m._is_residual = True` impostato nel costruttore dei blocchi),
> perché `nn.Linear` da solo non "sa" se scrive nel residuo. Lo gestiamo
> nell'implementazione.

---

## 7. Mini-esempio numerico end-to-end

Modello a **N = 6** layer ⇒ `2N = 12` ⇒ fattore `1/√12 ≈ 0.289`.

- Pesi normali: `std = 0.02`
- Pesi residuali: `std = 0.02 × 0.289 ≈ 0.0058`

Partendo da un residuo con std ≈ 1 e immettendo 12 contributi:
- **SENZA scaling** (ogni contributo std ≈ 0.02·k): il residuo si gonfia di √12 ≈
  **3.46×** rispetto al caso ideale → segnale sbilanciato, layer profondi schiacciati.
- **CON scaling**: i 12 contributi sono ridotti di 0.289 ciascuno → l'accumulo
  resta ≈ **1.00×** → segnale stabile a qualsiasi profondità.

---

## 8. TL;DR

| Domanda | Risposta |
|---|---|
| Sintomo | 2 layer battono 6 → i modelli profondi non si allenano. |
| Causa | La varianza del residuo cresce come `√(2N)` → init non scala con la profondità. |
| Cura | Init `normal(0, 0.02)`; le **2 proiezioni residuali** (attn `out_proj`, FFN 2° Linear) a `normal(0, 0.02/√(2N))`. |
| Perché funziona | Il `1/√(2N)` annulla la crescita `√(2N)` → scala del residuo **costante** con la profondità. |
| Quando agisce | Solo all'**inizializzazione**: dà un punto di partenza ben condizionato. |
| Atteso | I modelli profondi tornano a migliorare → spazio per scendere sotto la PPL attuale. |

---

## 9. Come si chiama? E che tipo di tecnica è?

**Nome.** Non ha un singolo nome canonico universale; di solito si chiama
**residual scaling** (o *scaled residual initialization* / "trucco di init di
GPT-2"). Appartiene alla famiglia più ampia delle **inizializzazioni
variance-preserving / depth-aware**: stesso spirito di Xavier/Glorot e Kaiming/He
per le reti feed-forward, ma adattato alle connessioni residuali. La nostra è la
variante GPT-2: `1/√(2N)` sulle proiezioni residuali.

Parenti "con nome proprio" che attaccano lo stesso problema (varianza del residuo
in profondità):
- **Fixup** (Zhang et al. 2019) — init che allena ResNet *senza* normalization.
- **T-Fixup** (Huang et al. 2020) — versione per Transformer (niente warmup/LayerNorm).
- **ReZero** (Bachlechner et al. 2020) — scalare *apprendibile* sul ramo residuale,
  inizializzato a 0.
- **DeepNorm/DeepNet** (Wang et al. 2022) — init+norm per transformer molto profondi.

**Categoria: è inizializzazione — NON normalization, NON regularization.**
Sono tre cose distinte, con scopi diversi:

| | Cosa fa | Quando agisce | Scopo |
|---|---|---|---|
| **Initialization** (questa) | sceglie i **valori di partenza** dei pesi | **solo a t=0**, poi sparisce | **trainabilità** (gradienti sani, segnale stabile) |
| **Normalization** (LayerNorm, BatchNorm) | ri-scala/ri-centra le **attivazioni** | a **ogni** forward, train *e* inference | stabilità del segnale a runtime |
| **Regularization** (dropout, weight decay, early stop) | vincola/disturba il modello | durante il **training** | **generalizzazione** (meno overfitting) |

- **Non è normalization**: non aggiunge nessun layer né operazione a runtime; non
  tocca le attivazioni nel forward. Fissa solo i pesi iniziali, poi il training li
  muove liberamente. (Il LayerNorm del modello resta lì, è un'altra cosa.)
- **Non è regularization**: non cerca di ridurre l'overfitting, non aggiunge
  rumore, non penalizza i pesi, non toglie capacità. Il suo scopo è l'**opposto**
  del dropout — far allenare *meglio* i modelli (specie profondi), non frenarli.

> In una frase: è **inizializzazione depth-aware**, agisce solo sul *punto di
> partenza* per condizionare l'**ottimizzazione**. Per questo può far scendere la
> PPL senza "costare" capacità, a differenza di una tecnica di regularization.

---

## Appendice A — La legge statistica: perché le varianze si sommano (e lo std cresce come √n)

Tutto il documento si appoggia a una frase: *"sommando quantità indipendenti, le
varianze si sommano"*. Qui la sviscero, perché è il cuore del perché serve `1/√(2N)`.

### A.1 La legge

Per variabili casuali `X₁, …, Xₙ`:

```
   Var(X₁ + … + Xₙ) = Σ Var(Xᵢ)  +  Σ Cov(Xᵢ, Xⱼ)
                       └─ termini  ┘   └─ termini incrociati (i≠j) ┘
```

Se le variabili sono **indipendenti** (o anche solo *scorrelate*), tutte le
covarianze `Cov(Xᵢ, Xⱼ)` sono **zero**, e resta:

```
   Var(X₁ + … + Xₙ) = Var(X₁) + … + Var(Xₙ)        ← "le varianze si sommano"
```

Se in più i contributi sono **identici** (stessa varianza `σ²`) e a media zero:

```
   Var(somma) = n · σ²        →        std(somma) = √n · σ
```

**Lo std cresce come √n, non come n.** Questo `√` è tutto.

### A.2 Perché le covarianze spariscono

`Cov(X, Y) = E[(X−μ_X)(Y−μ_Y)]` misura se X e Y "oscillano insieme". Se sono
indipendenti, sapere X non dice nulla su Y, l'aspettazione si fattorizza e
`Cov(X, Y) = 0`. Tradotto: le fluttuazioni **non si rinforzano sistematicamente**.

### A.3 L'intuizione: la passeggiata dell'ubriaco 🍷

Esempio canonico. Fai `n` passi di **±1** a caso (testa/croce), ognuno con media 0
e varianza 1. La tua **posizione finale** è la somma dei passi.

```
   Var(posizione) = n · 1 = n        →        std(posizione) = √n
```

Dopo **100** passi casuali sei tipicamente a **~10** passi dall'origine, **non a 100**.

Perché? Perché i passi vanno in direzioni diverse e **si cancellano in parte**:

```
   Se i contributi fossero TUTTI allineati (perfettamente correlati):
        |somma| ≈ n · s          (crescita LINEARE)
   Se sono indipendenti (direzioni casuali → cancellazione parziale):
        std    ≈ √n · s          (crescita a RADICE)
```

Il `√n` è **la firma dell'indipendenza**: gli std si sommano "in quadratura"
(`√(a² + b²)`), non linearmente. Il residual stream è esattamente questo: ogni
blocco aggiunge un "passo casuale" all'autostrada; dopo **2N** passi lo
spostamento accumulato ha std `√(2N) · s`. Ecco da dove esce il `√(2N)`.

### A.4 Esempio numerico

`n` contributi indipendenti, ognuno con std `s = 1`:

| n (contributi) | Var = n·s² | std = √n |
|:---:|:---:|:---:|
| 1  | 1  | 1.00 |
| 4  | 4  | 2.00 |
| 9  | 9  | 3.00 |
| 12 | 12 | 3.46 |  ← il nostro caso a 6 layer (2N=12)
| 16 | 16 | 4.00 |

Per riportare lo std a 1 con `n = 12`, basta ridurre ogni contributo di `1/√12 ≈
0.289`: `√12 · (1 · 0.289) = 1`. **È esattamente la Leva 1.**

### A.5 Bonus: perché lo std dell'output di un Linear ∝ std dei pesi

Serve a chiudere il cerchio ("scalo lo std dei pesi ⇒ scalo lo std del contributo
nel residuo"). Un neurone calcola `y = Σⱼ wⱼ·xⱼ` su `d` ingressi (fan-in `d`). Con
`w` e `x` indipendenti e a media zero, applicando di nuovo la legge della somma:

```
   Var(y) = Σⱼ Var(wⱼ·xⱼ) = d · Var(w) · Var(x)
   std(y) = √d · std(w) · std(x)
```

Quindi **`std(y)` è proporzionale a `std(w)`**: dimezzo lo std dei pesi → dimezzo
lo std dell'output. Moltiplicare lo std dei pesi residuali per `1/√(2N)` riduce di
`1/√(2N)` il contributo immesso nel residuo — chiudendo il discorso.
*(È anche la base delle init di Xavier/He: scelgono `Var(w) = 1/d` per tenere
`Var(y) = Var(x)`, cioè per non gonfiare né schiacciare il segnale strato per strato.)*

### A.6 Le assunzioni (e perché è "solo" all'init)

Nella rete vera i contributi **non** sono perfettamente indipendenti né a media
esattamente zero (ci sono LayerNorm, non-linearità, correlazioni apprese). Quindi
`√(2N)` è un'**approssimazione valida all'inizializzazione**, non una legge esatta
durante tutto il training. Ma cattura la dinamica dominante "a t=0", ed è proprio
all'init che ci serve un punto di partenza ben condizionato — il training poi
aggiusta il resto.

---

## Appendice B — Perché lo scaling va SOLO sulle proiezioni residuali (e non sulle altre)

La cura `1/√(2N)` si applica **solo** a `out_proj` e al 2° `Linear` della FFN, non
a `w_q/w_k/w_v`, non al 1° `Linear` della FFN, non a `lm_head`, non agli embedding.
Ecco perché — ed è una distinzione **"writer vs reader"**.

### B.1 Solo una cosa si accumula: il residual stream

Riprendi la legge dell'Appendice A: lo std cresce come `√(numero di addizioni)`.
Ma cosa viene **addizionato 2N volte**? **Solo il residual stream.** È *l'unica*
quantità sommata ripetutamente attraverso i blocchi. Quindi è l'unica che cresce
come `√(2N)`, e l'unica da tenere a bada.

Di conseguenza, devono essere scalate **solo le proiezioni il cui output viene
sommato nel residuo** — le chiamiamo **"writer"**. Le altre non si accumulano:
scalarle non risolverebbe nulla (non toccano la quantità che esplode) e farebbe
solo danni (vedi B.3).

### B.2 Writer vs reader: chi è chi nel nostro blocco

```
        ┌──────────────── residual stream (QUESTO accumula) ────────────────┐
   x ───●──────────────────────────────────────────●───────────────────────►
        │                                           │
        │   LN → [w_q,w_k,w_v] → attn → out_proj ──(+)   LN → [Lin1]→GELU→[Lin2] ──(+)
        │         └── READER ──┘      └ WRITER ┘              └READER┘   └WRITER┘
```

| Strato | Ruolo | Scrive nel residuo? | Init |
|---|---|:---:|---|
| `w_q`, `w_k`, `w_v` | **leggono** lo stream (normalizzato) per calcolare l'attention | No | `normal(0, 0.02)` |
| `out_proj` | **scrive** il risultato dell'attention nel residuo | **Sì** | `normal(0, 0.02/√(2N))` |
| FFN `Linear1` (d→hidden) | **legge/espande** per calcolare la FFN | No | `normal(0, 0.02)` |
| FFN `Linear2` (hidden→d) | **scrive** il risultato della FFN nel residuo | **Sì** | `normal(0, 0.02/√(2N))` |
| `token_embed`, `pos_embed` | **impostano** lo stato iniziale `x₀` (sommato una volta) | Imposta, non accumula | `normal(0, 0.02)` |
| `lm_head` | **legge** lo stream finale → logit (una volta) | No | `normal(0, 0.02)` |

### B.3 Perché i "reader" NON vanno scalati

Due motivi, entrambi importanti:

1. **Non contribuiscono all'accumulo.** L'output di `w_q/w_k/w_v` finisce dentro il
   softmax dell'attention; quello di `Linear1` finisce nella GELU. **Nessuno dei
   due viene sommato nel residuo**: ci arriva solo *dopo*, passando per il writer
   (`out_proj` / `Linear2`). Scalare un reader non cambia di una virgola la
   varianza del residual stream.

2. **Scalarli farebbe danno.** Ridurre `w_q/w_k/w_v` o `Linear1` significa
   **indebolire il calcolo interno** del blocco: attention scores quasi piatti,
   attivazioni FFN minuscole → ogni blocco calcola una funzione **degenere/debole**.
   Perderesti capacità rappresentativa senza toccare il problema vero (l'accumulo).

> Divisione del lavoro: vogliamo che ogni blocco dia una **spinta piccola e
> controllata** all'autostrada (→ writer scalati), ma che il suo **calcolo interno
> sia a piena potenza** (→ reader a `0.02`), così la spinta è *significativa*.
> Scalare tutto darebbe spinte piccole **e** calcolo debole: il peggio.

### B.4 Il ruolo del Pre-LayerNorm (perché i reader sono "protetti")

C'è un motivo in più, strutturale. Usiamo il **Pre-LN**: prima di ogni sotto-modulo
c'è un `LayerNorm` che **ri-normalizza** l'input.

```
   x → LayerNorm → (w_q,w_k,w_v / Linear1) → ...
        └ riporta a scala unitaria ┘
```

Quindi i reader **non vedono mai** il residuo "gonfio": vedono sempre un input a
scala ~1, qualunque sia la profondità. Il loro output dipende solo dal proprio
init (`0.02`) ed è già ben condizionato — **non serve** renderli depth-aware. È
**solo** la *scrittura di ritorno* nel residuo (che bypassa il LayerNorm e si
somma all'autostrada non normalizzata) a dover essere scalata.

### B.5 E embedding / lm_head?

- Gli **embedding** *impostano* `x₀` (lo stato iniziale): è **una** addizione, non
  2N → niente accumulo da domare → init normale `0.02`.
- `lm_head` **legge** lo stream finale una volta sola per produrre i logit → non
  scrive nel residuo → init normale `0.02` (e se c'è weight tying, eredita la scala
  dell'embedding, come spiegato nel corpo del documento).

**In una riga:** scali ciò che **scrive ripetutamente** nell'unica quantità che si
accumula. Tutto il resto — chi legge, chi imposta, chi calcola — resta a `0.02`.

---

## Appendice C — L'init è casuale, e perché un setting "una tantum" conta per tutto il training

Tre dubbi naturali: *perché chiamiamo i pesi "variabili casuali"? cosa c'entra la
probabilità? e se setto la scala solo all'inizio, come fa a valere per tutto il
training?*

### C.1 I pesi sono variabili casuali perché l'init li *estrae a sorte*

All'inizializzazione **non scegliamo** i pesi: `normal(0, 0.02)` significa "pesca
ogni peso da una gaussiana". Quindi l'init **è un processo casuale** e i pesi
appena nati sono un **campione** da una distribuzione.

Ragioniamo sulle distribuzioni perché vogliamo prevedere il comportamento
**tipico** che la ricetta produce — così funziona per un'estrazione *qualsiasi*,
non per una fortunata. Quando scriviamo "la varianza del residuo è `√(2N)·s`"
stiamo descrivendo cosa fa l'init **in tipico**, non il valore di una rete singola.

Doppia lettura della "varianza", che qui coincidono:
- **tra inizializzazioni diverse** (ripeto l'init tante volte → distribuzione);
- **tra i tanti pesi/attivazioni di UNA rete** (gli strati sono larghi → per la
  legge dei grandi numeri la dispersione empirica ≈ quella teorica).

→ L'analisi probabilistica descrive **anche il tuo modello concreto**, non solo
un'astrazione.

### C.2 "Setto solo all'inizio": è un punto di partenza, non un invariante

Attenzione all'equivoco: l'init **non** mantiene la std scalata per tutto il
training — i pesi cambiano, eccome. L'init fa una cosa diversa: ti mette in un
**punto di partenza ben condizionato**, e questo **gate l'avvio**.

```
   init depth-unaware  →  passo 0: gradienti esplodono/svaniscono in profondità
                       →  strati profondi senza segnale / divergenza
                       →  NON recuperi piu' (l'ottimizzatore parte gia' rotto)

   residual scaling    →  passo 0: gradienti sani e bilanciati su tutta la profondita'
                       →  SGD parte bene  →  training stabile verso una buona soluzione
```

Perché un setting "una volta sola" conta per tutto il viaggio:
- 🎸 **Accordare la chitarra**: la accordi *una volta*. L'accordatura non tiene
  ferme le note durante il pezzo (lo fanno le dita), ma partire **scordati** rovina
  tutta l'esecuzione. L'init è l'accordatura.
- 🏔️ **Dove parti sulla montagna**: scegli il punto di partenza una volta, ma
  decide se la discesa è *possibile*. Su un dirupo (gradienti esplosivi) o su un
  altopiano piatto (gradienti nulli) non scendi mai; in una valle dolce
  l'ottimizzatore fa il resto.

### C.3 Chi mantiene la scala *durante* il training? Il LayerNorm

L'init dà il buon **punto di partenza** (una volta). A tenere la scala sotto
controllo **a ogni passo** è il **LayerNorm** (una *normalization*, che agisce a
runtime, ri-normalizzando l'input di ogni sotto-modulo a ogni forward).

```
   Init       = buon PUNTO DI PARTENZA      (una volta)        → abilita il training
   LayerNorm  = mantiene la scala A RUNTIME (ogni forward)     → la tiene per sempre
```

Ecco perché (richiamo alla sezione 9) l'init è *initialization* e non
*normalization*: non deve "tenere" la std per sempre, deve solo **non rompere nulla
al passo 0** e lasciar fluire i gradienti. Il resto — ottimizzatore, LayerNorm,
pesi appresi — tiene in piedi la baracca.

---

## Riferimenti
- Radford et al., **"Language Models are Unsupervised Multitask Learners"** (GPT-2,
  2019): introduce lo scaling `1/√N` dei layer residuali all'init.
- Vaswani et al., **"Attention Is All You Need"** (2017): architettura e residual
  connections.
- Zhang et al., **"Fixup Initialization"** (2019); Huang et al., **"T-Fixup"**
  (2020); Bachlechner et al., **"ReZero is All You Need"** (2020); Wang et al.,
  **"DeepNet"** (2022): famiglia di tecniche che controllano la varianza del
  residuo in profondità.
- Glorot & Bengio (Xavier, 2010); He et al. (Kaiming, 2015): le inizializzazioni
  variance-preserving classiche per reti non-residuali.
