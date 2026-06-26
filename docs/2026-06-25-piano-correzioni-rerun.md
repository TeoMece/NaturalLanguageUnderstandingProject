# Piano correzioni + re-run (per un lavoro coerente su tutte le parti)

Master checklist di TUTTO ciò che resta da sistemare. Il **codice è già pronto**: qui si
tratta di (a) una piccola aggiunta infrastrutturale (multi-seed), (b) ri-girare gli
esperimenti in modo coerente, (c) aggiornare report/recap. Scelta utente: **multi-seed
su OGNI config di ricerca** (opzione "fedele al lab", media ± std).

---

## 0. Prerequisito multi-seed + media ± std — ✅ FATTO (branch feat/rerun-multiseed)

- [x] **Run su N seed**: `sweep: { experiment.seed: [42,1,2,3,4] }` → run distinte
  (`..._seed42`, ...) e il seed viene usato (`set_seed(exp.seed)` in tutti gli experiment).
  Verificato. Nessuna modifica di codice necessaria.
- [x] **Aggregatore media ± std**: `seed_summary.py` (standalone, come gap_report) —
  raggruppa per config (seed rimosso dal nome) e stampa `media ± std` (PPL per LM, slot
  F1 / intent acc in % per NLU). +2 test.

**SCOPE DECISO (2026-06-25): multi-seed solo su VINCITORI di ogni step + FINALI.** La
ricerca larga resta a **seed singolo** (per scegliere i vincitori); poi si ripete ×5
seed solo la config vincente di ogni step e il modello finale → media ± std dove conta.
Costo gestibile (decine di run), standard.

### Ricetta multi-seed (per ogni vincitore / finale)
1. crea una config con gli iperparametri vincenti + blocco:
   `sweep: { experiment.seed: [42, 1, 2, 3, 4] }`  (per il finale aggiungi `mode: final`).
2. `python run*.py sweep --config <quella_config>`  → 5 run `..._seed42`, `..._seed1`, ...
3. `python seed_summary.py --runs <runs_dir>`  → tabella **media ± std**.
Nel report: i numeri chiave (finale, e i confronti decisivi) vanno come media ± std;
la tabella di ricerca resta a seed singolo, dichiarandolo.

---

> **Tutto il lavoro è su un unico branch (la "versione corretta").** Il multi-seed
> (media±std) riguarda **SOLO la Parte 2** (lo chiede il lab 5/NLU); la **Parte 1 resta
> a seed singolo** (il lab 4/LM non lo richiede).

## 1. Parte 1.A (LM, GPT2 da zero) — seed singolo

- [x] **Portare sul branch i config mancanti** `01b_ff.yaml`, `01c_heads.yaml` e
  `00_starting_point.yaml` — ricreati su `final` (commit 88a8cca). Caricano: ff 3 run,
  heads 3 run, starting_point 1 run.
- [ ] **Completare la ricerca iperparametri**: sweep `ff_mult` e `n_heads` (finora solo
  `d_model × num_layers`).
- [x] ~~Starting point lr 0.1~~ **RIMOSSO (2026-06-25)**: il lab definisce lo Step 0 come
  la **baseline con ricerca del lr** (`00_baseline`); il delta onesto è baseline→finale.
  Config `00_starting_point.yaml` eliminati (1.A e 2.A).
- [ ] **Aggiornare** report LM (Tab.1) e recap coi nuovi numeri (delta baseline→finale).
- [x] **Onestà dropout**: `p0.1` è within-noise, non un "real gain" (vero regolarizzatore = tying).

## 2. Parte 1.B (LoRA) — seed singolo

- [ ] **Starting point onesto**: GPT2 **pre-addestrato SENZA adapter** (zero-shot) — mostra
  quanto aggiunge LoRA. (Forma da decidere.)
- [ ] **Aggiornare** recap_b + parte LM del report col delta.

## 3. Parte 2.A (NLU, GPT2 da zero) — multi-seed sui vincitori/finali

- [ ] **Ricerca** (seed singolo): lr → arch → ff → heads → dropout (config già qui).
- [ ] **Multi-seed (×5)** sui **vincitori di ogni step** e sul **finale** → media ± std
  (`experiment.seed` sweep + `seed_summary.py`).
- [ ] **Aggiornare** recap_nlu (media±std) + scrivere la parte 2.A del report NLU.
- [x] onestà dropout: già a posto. (starting_point lr 0.1 RIMOSSO 2026-06-25.)

## 4. Parte 2.B (NLU, GPT2 + BERT pre-addestrati) — branch `feat/part2-nlu` — DA GIRARE

- [ ] **Sweep gpt2 + bert** (lr {5e-5,3e-5,2e-5}) **con multi-seed** → media ± std.
- [ ] **Finalize entrambi** i modelli (per il confronto encoder vs decoder) → test.
- [ ] **Export** `NLU/part_B` (pesi grandi → **zip di consegna, non git**).
- [ ] **Scrivere** `reports/recap_nlu_b.md` (confronto GPT2 vs BERT) + parte 2.B del report.

## 5. Report finali (consegna)

- [ ] **Report LM** (1.A + 1.B): max 1 pagina, template INTERSPEECH, coi numeri
  e i delta onesti (baseline→finale). Dichiarare uso AI; nome/matricola.
- [ ] **Dichiarare nel report LM lo scheduler warmup+cosine come NOSTRA scelta** (non
  richiesta dal lab): motivazione = stabilità del LM da-zero nelle prime iterazioni +
  rifinitura finale; contributo quantificato dall'ablazione `00b_no_scheduler` (gemello
  del baseline: scheduler OFF sui 3 lr, subito dopo il baseline) → mostra anche se il
  vantaggio dipende dall'lr. Scheduler ON come ricetta standard per il resto della catena.
- [ ] **Dichiarare nel report LM il cambio di pazienza allo step dropout**: per dare al
  modello regolarizzato (rumore del dropout → curva di valid più rumorosa, convergenza più
  lenta) il tempo di convergere, allo step `02_dropout`/`03_weight_tying` abbiamo alzato
  `patience 3→6` ed `epochs 30→50`, e aggiunto `p=0.0` come riferimento NO-dropout alle
  STESSE condizioni. Risultato: **`p=0.1` è il migliore (34.55)**, batte il riferimento
  no-dropout `p=0.0` (**34.84**) di **0.29** a parità di condizioni → con la nuova arch più
  grande (56M) il dropout LEGGERO aiuta davvero (U-shape: 0.2→35.57, 0.3→36.07, 0.4→38.34).
  Questo è un gain REALE (sopra il rumore ~0.05), diverso dal vecchio run (29M) dove era
  within-noise. Nota meccanica: `p=0.0` dà 34.84 vs 34.77 dello step heads (epochs 30) per
  via dello scheduler (`total_steps=epochs×batch`, `epochs 30→50` allunga il cosine), non
  per il dropout — il confronto onesto è p=0.1 vs p=0.0 a epochs=50.
- [ ] **Dichiarare nel report LM l'init con residual scaling (Leva 1, GPT-2)**: le
  proiezioni residuali (`out_proj` attn + 2° Linear FFN) sono init con
  `std = 0.02/sqrt(N)`, `N = 2·num_layers` (= 1/√(2·num_layers) × 0.02) per tenere
  costante la varianza del residual stream con la profondità. È il fix che ha risolto
  la divergenza della std del baseline iniziale. **Motivazione da dichiarare**: serve a
  rendere ALLENABILI/COMPETITIVE le architetture più grandi/profonde — senza, i modelli
  con più layer divergerebbero o renderebbero peggio, falsando lo sweep d_model×num_layers
  a favore dei piccoli (con il fix 512×4 / 384×6 ecc. sono trainabili ad armi pari).
  Vedi `docs/leva1-init-residual-scaling.md`.
- [ ] **Report NLU** (2.A + 2.B): max 1 pagina, encoder vs decoder, media±std.
- [ ] **Zip di consegna** con i `bin/` (pesi) inclusi (gitignore non li esclude dallo zip).

---

## Ordine consigliato (per velocità e coerenza)

1. **Codice multi-seed** (prerequisito) — lo aggiungo io in tutte le pipeline.
2. **Parte 2.B** prima (è l'unica MAI girata): sweep + multi-seed + finalize + export.
3. **Re-run multi-seed** di 2.A (config pronti), poi 1.A (+ ff/heads + starting_point), poi 1.B.
4. **Aggiornare recap/report** man mano.
5. **Scrivere i due report** finali.

## 6. Audit copertura parametri del lab (verificato 2026-06-25)

- **1.A**: lr ✅, d_model ✅, num_layers ✅, dropout(4 punti) ✅, weight tying ✅;
  **mancano `n_heads` e `ff_dim`** → `01b_ff` / `01c_heads` (pronti, da girare). Sono le
  uniche modifiche del lab non ancora toccate in tutto il progetto.
- **1.B**: rank ✅, alpha ✅.
- **2.A**: lr/d_model/num_layers/n_heads/ff_dim ✅ + dropout-prima-delle-teste ✅ → COMPLETO.
- **2.B**: gpt2 ✅, bert ✅, lr ✅ (gpt2-medium/bert-large opzionali).
- Vincolo coerenza: `n_heads` deve dividere `d_model` (`[2,4,8]` ok per 256 e 384); gli
  sweep ff/heads tengono fissa la miglior arch del passo precedente ("one at a time").

## 7. Analisi extra per i report (originalità) — TENERE MINIMALE

Decisione (2026-06-25): stare su una **base pulita** (il report è max 1 pagina).

- [ ] **Gap di overfitting** per run (`gap_report.py` + figura): già pronto, l'unica
  figura extra che teniamo (è "carina" e non conta nel limite pagina). Le run con
  dropout sono già marcate (gap inattendibile), quindi è onesto così com'è.
- [x] ~~Correlazione overfitting ↔ dimensione modello con/senza dropout + fix gap
  eval-mode~~ → **SCARTATO**: troppo per una pagina, rischia di sporcare il lavoro.
  (Se mai servisse: richiederebbe griglia fattoriale size×dropout + train PPL in
  eval-mode per un gap onesto. Non lo facciamo.)

## Cosa NON va rifatto
- Le **pipeline/codice** (data, modelli, train, export, CLI, test): fatte e testate (58 test verdi).
- La **metodologia** (dev stratificato, selezione su slot F1 / valid PPL, test sigillato): ok.
