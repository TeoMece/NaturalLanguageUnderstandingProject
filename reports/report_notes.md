# Appunti per il report finale (LM) — idee chiare prima di scrivere

> Promemoria operativo: cosa dire, come frammarlo, cosa fare *prima* di scrivere.
> Il report va in inglese, template IEEE, **max 1 pagina** (escluse tabelle/figure/ref),
> più peso alla parte che vale di più, e **va dichiarato l'uso di AI**.

---

## 0. La cosa più importante: NON è una gara di PPL

- L'unico requisito sul numero è **PPL < 250** (soglia, non classifica). La sfondiamo
  largamente (Part A ~34 valid; Part B ~21).
- Il voto è **Code + Report (80%) + Q&A (20%)**. Il numero in sé non è il voto: contano
  **metodologia, chiarezza del report, originalità, comprensione**.
- Strategia scelta: **privilegiare la discussione dei limiti e della metodologia**
  rispetto alla caccia al decimale. È ciò che dimostra padronanza e alza il voto.
- Il nostro punto di forza è capire *perché* escono quei numeri → sfruttarlo nel report
  e alla Q&A.

---

## 1. Cosa raccontare — Parte 1.A (GPT2 from scratch)

- **Storia incrementale** (una modifica alla volta), con la tabella di `partA_summary`:
  lr → architettura → dropout → weight tying.
- **Best**: `d384/l2` + dropout p0.1 + weight tying, lr 1e-3 → **valid 34.18**
  (→ test dopo `finalize`).
- **Scoperte da evidenziare**:
  - *bigger ≠ better* su PTB: il modello grande/profondo (`d512/l6`) è tra i **peggiori**
    (40.97). Su un corpus piccolo i modelli grandi da zero non rendono.
  - **weight tying** è la leva singola più forte (36.24 → 34.18) e toglie ~19M parametri.
  - **dropout**: aiuta solo leggero (p0.1); oltre, peggiora.

### 1.1 Il punto "originalità/metodologia" forte: l'init
- I risultati si sono **ribaltati** dopo aver corretto l'inizializzazione (la **std del
  residuo divergeva con la profondità** senza lo scaling `1/√(2·num_layers)` — "Lever 1";
  + fix embedding/tying). Pre-fix vinceva il modello grande e il tying era catastrofico
  (69.68); post-fix vince `d384/l2` e il tying è la svolta (34.18).
- Da raccontare come **finding metodologico** (confronto pre/post-fix, archiviato in
  `reports_pre_fix/`). Dimostra che sappiamo *perché* l'init conta.

### 1.2 grad_clip come regolarizzatore (finding, NON "colpa")
- Il nostro `grad_clip=1.0` agisce da **regolarizzatore**: passi piccoli → fitting più
  conservativo → meno overfitting → ottimo di dropout **basso** (p0.1). Con clip più
  largo (es. 5) il modello fitta di più → un dropout più alto (0.3) aiuta.
- **Framing nel report**: scelta *analizzata*, non errore. "Il dropout ottimale non è
  universale: dipende dall'interazione clip ↔ dropout ↔ overfitting." Spiega perché
  un'altra implementazione dello stesso esercizio trova un dropout diverso — **non è
  fortuna**. (Dettaglio in `docs/2026-06-15-lm-pipeline-part1a-design.md` §13.)

---

## 2. Cosa raccontare — Parte 1.B (LoRA su GPT2 pre-addestrato)

- **LoRA fatto a mano** (no PEFT) su Q/K/V, **solo adapter allenati** (~442k / 124M,
  **~0.35%**), backbone congelato. `ΔW = (α/r)·B·A`, `A~N(0,0.02)`, `B=0` (no-op a init).
- **Trend**: PPL **monotòna decrescente col rank** (rank16 best); a rank16, **α più basso
  aiuta** (α16 > α32; α8 in corso, come il compagno).
- **Best (finora)**: rank16/α16 → valid 21.48 (→ test dopo `finalize`; α8 potrebbe battere).
- **Punto chiave del progetto**: Part B **batte** Part A col **pre-training** → è la
  risposta al limite di 1.A ("PTB è data-limited, la profondità rende solo con dati/
  pre-training").
- Possibile **analisi/originalità**: magnitudo dell'aggiornamento LoRA per layer, oppure
  l'analisi del gap di overfitting (`gap_report.py --runs runs_b`); in B il gap è
  **affidabile** (niente dropout). Dettagli concettuali: appendici A.1–A.8 in
  `docs/2026-06-22-part1b-lora.md`.

---

## 3. Metodologia trasversale da dichiarare

- **valid vs test**: gli sweep girano in `mode: dev` (valid, test sigillato); il numero
  ufficiale arriva da **`finalize`** (test). NB: su PTB il **test è ~2 PPL sotto il
  valid** → i numeri "veri" del report saranno più bassi di quelli degli sweep.
- **gap di generalizzazione** (`gap_report.py`): usarlo per giustificare le scelte di
  regolarizzazione (caveat: il gap è affidabile solo sulle run senza dropout in 1.A).
- **Riproducibilità**: seed fissi, git hash, config archiviata per run.

---

## 4. DA FARE prima di scrivere (poco lavoro, alto valore)

- [ ] **`finalize` Part A** (la best `d384/l2+p0.1+tying`) → **test PPL** ufficiale.
- [ ] **`finalize` Part B** (la best dopo lo sweep α) → **test PPL** ufficiale.
- [ ] Finire la run **α8** (`03_alphaEight.yaml`) e `aggregate` → vincitore di B.
- [ ] *(opzionale, se avanza tempo)* una run **ablazione `grad_clip=5`** su Part A:
      proverebbe l'interazione clip↔dropout *e* probabilmente abbasserebbe il numero.

---

## 5. Promemoria sul formato

- Max **1 pagina** (ref/tabelle/figure escluse) — tagliare senza pietà.
- Più peso alla parte che **vale di più** (controllare la suddivisione punti nel
  template/assegnazione).
- **Dichiarare l'uso di AI** (come fa anche il report di riferimento).
- Tabelle pronte: `partA_summary.tex` / `partB_summary.tex` (booktabs).
