# Piano correzioni + re-run (per un lavoro coerente su tutte le parti)

Master checklist di TUTTO ciò che resta da sistemare. Il **codice è già pronto**: qui si
tratta di (a) una piccola aggiunta infrastrutturale (multi-seed), (b) ri-girare gli
esperimenti in modo coerente, (c) aggiornare report/recap. Scelta utente: **multi-seed
su OGNI config di ricerca** (opzione "fedele al lab", media ± std).

---

## 0. Prerequisito (codice, una volta sola): multi-seed + media ± std

Senza questo non si può ri-girare "in modo affidabile". Serve in tutte le pipeline
(`lm_pipeline`, `lm_pipeline_b`, `nlu_pipeline`, `nlu_pipeline_b`):

- [ ] **Run su N seed**: il sweep supporta già le chiavi dotted → `sweep: experiment.seed:
  [42, 1, 2, 3, 4]`. Va solo verificato che `experiment.seed` finisca nel nome run
  (suffisso `__seed1`, ...) per non sovrascrivere.
- [ ] **Aggregatore media ± std**: nuovo — raggruppa le run che condividono la config
  *tranne il seed* e calcola media e deviazione standard della metrica (PPL per LM,
  slot F1 / intent acc per NLU). Output: una tabella `X.XX ± Y.YY`.
- [ ] **N = 5 seed** (es. [42,1,2,3,4]) come default.

> Nota costo: con multi-seed ogni step si moltiplica ×5. ATIS/PTB sono piccoli e le run
> brevi, ma fai i conti (Parte 2.A da sola: 19 config × 5 ≈ 95 run). Su V100 fattibile.

---

## 1. Parte 1.A (LM, GPT2 da zero) — branch `feat/residual-init`

- [ ] **Completare la ricerca iperparametri**: girare gli sweep mancanti `ff_mult`
  (`configs/experiments/01b_ff.yaml`) e `n_heads` (`01c_heads.yaml`) — **config già
  pronti** sul branch. (Finora swippati solo `d_model × num_layers`.)
- [ ] **Starting point onesto**: baseline pre-tuning con **lr ingenuo 0.1** (default del
  lab) → delta baseline→finale onesto. (Config `starting_point` da creare, come per 2.A.)
- [ ] **Multi-seed** su tutte le config di ricerca → media ± std.
- [ ] **Aggiornare** `LM/report/report.tex` (Tab.1) e `reports/recap.md` coi nuovi numeri
  (media±std, eventuali nuovi vincitori, delta dello starting_point).
- [x] **Onestà dropout**: già corretto (`p0.1` è within-noise, non un "real gain";
  il regolarizzatore vero è il weight tying).

## 2. Parte 1.B (LoRA) — branch `feat/part1b-lora`

- [ ] **Starting point onesto**: il GPT2 **pre-addestrato valutato SENZA adapter**
  (zero-shot) — mostra quanto aggiunge LoRA; in alternativa lr ingenuo. (Forma da
  decidere.)
- [ ] **Multi-seed** sulle run di rank/alpha → media ± std (LoRA è veloce, costo basso).
- [ ] **Aggiornare** `reports/recap_b.md` e la parte LM del report col delta + std.

## 3. Parte 2.A (NLU, GPT2 da zero) — branch `feat/part2-nlu`

- [ ] **Multi-seed su TUTTE le 19 config di ricerca** (lr 3, arch 6, ff 3, heads 3,
  dropout 4) → media ± std. (Opzione 2 scelta.) ATIS è piccolo → run brevi.
- [ ] **Aggiornare** `reports/recap_nlu.md` coi numeri media±std; verificare che il
  "tutto converge sul baseline" regga (probabile: le differenze sono dentro lo std).
- [ ] **Scrivere** la parte 2.A del report NLU.
- [x] starting_point lr 0.1: già girato ma **da ignorare** (non è uno step della catena).
- [x] Onestà dropout: già a posto (none è il migliore).

## 4. Parte 2.B (NLU, GPT2 + BERT pre-addestrati) — branch `feat/part2-nlu` — DA GIRARE

- [ ] **Sweep gpt2 + bert** (lr {5e-5,3e-5,2e-5}) **con multi-seed** → media ± std.
- [ ] **Finalize entrambi** i modelli (per il confronto encoder vs decoder) → test.
- [ ] **Export** `NLU/part_B` (pesi grandi → **zip di consegna, non git**).
- [ ] **Scrivere** `reports/recap_nlu_b.md` (confronto GPT2 vs BERT) + parte 2.B del report.

## 5. Report finali (consegna)

- [ ] **Report LM** (1.A + 1.B): max 1 pagina, template INTERSPEECH, coi numeri
  media±std e i delta onesti (starting_point). Dichiarare uso AI; nome/matricola.
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

## 7. Analisi extra per i report (originalità)

- [ ] **Gap di overfitting** per run (`gap_report.py` + figura): già pronto, estendere
  alle altre parti.
- [ ] **Fix onesto del gap**: misurare il **train PPL/loss in EVAL mode (dropout off)** a
  fine training, così il gap `valid − train` non è falsato dal dropout attivo. Piccola
  aggiunta alla pipeline (un passaggio di eval sul train). Necessario per qualsiasi
  confronto with/without-dropout.
- [ ] **Correlazione overfitting ↔ dimensione modello, con/senza dropout** (1.A, LM):
  griglia fattoriale `d_model {256,384,512} × dropout {0, p}` (num_layers fisso) +
  script di plot "gap vs dimensione" con due linee (dropout on/off). Usa il gap onesto
  del punto sopra. Mostra: modelli più grandi overfittano di più; il dropout chiude il
  gap. Bella figura di originalità per il report LM.

## Cosa NON va rifatto
- Le **pipeline/codice** (data, modelli, train, export, CLI, test): fatte e testate (58 test verdi).
- La **metodologia** (dev stratificato, selezione su slot F1 / valid PPL, test sigillato): ok.
