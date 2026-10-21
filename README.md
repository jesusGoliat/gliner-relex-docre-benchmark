# Document-level relation extraction with GLiNER-Relex on Re-DocRED (low-resource, end-to-end)

Official evaluation record: code, protocol, metrics and final results.

## 1. Overview

**Objective.** Measure end-to-end document-level relation extraction (text → entities → relations) with **GLiNER-Relex** when only **N = 10** or **N = 50** labeled training documents are available, and measure the effect of an LLM judge that re-decides only the extractor's *uncertain* candidates.

**Benchmark.** Re-DocRED (Tan et al., EMNLP 2022): 96 Wikidata relations, 6 entity types (PER, ORG, LOC, TIME, NUM, MISC); train 3 053 / dev 500 / test 500 documents. Low-resource training sets are the 5 published subsets per N (`data/subsets/redocred/train_N{10,50}_s{0..4}.json`). All reported numbers are on the full **test split (500 documents)**.

**Pipelines in the official evaluation**

| Id | Pipeline | LLM at inference | Training data |
|---|---|---|---|
| **Z0** | GLiNER-Relex large-v1.0, zero-shot | no | none |
| **B0** | GLiNER-Relex fine-tuned on the N gold documents | no | N gold documents |
| **J0** | B0 + LLM judge on the uncertain candidates | yes (`openai/gpt-oss-120b`) | N gold documents (also used as few-shot examples for the judge) |

Because GLiNER-Relex does not output coreference, entities are built from predicted mentions (rule-based clustering), so every number below is **end-to-end**: it includes mention detection, entity typing, entity grouping and relation classification. These numbers are not comparable with published figures that assume gold entities.

## 2. Evaluated Pipelines

Shared components (`src/`):

1. **Extractor** (`modal/relex_modal.py`): `knowledgator/gliner-relex-large-v1.0` through `gliner==0.2.29` (`UniEncoderSpanRelexGLiNER`). Each document is passed whole as one text (tokens joined by spaces; the model's `max_len` is 2 048 words, the longest dev document has 481 tokens and the longest test document 510). It returns entity spans (character offsets, label, score) and relations between spans (`head`, `tail`, relation name, score) for the 96 relation names (`data/external/.../rel_info.json`).
2. **Mention → entity aggregation** (`relex_post.py`, `ent_cluster.py`, `ent_build.py`): spans are converted to Re-DocRED token positions (`relex_io.span_tokens`); overlapping mentions inside a sentence are flattened (highest score wins, then longest); mentions are grouped into entities with rule-based coreference (normalized name, person aliases, acronyms/containment, dates, numbers); the entity type is the score-weighted argmax. The entity-level score of a relation is **S[h, t, r] = max over mention pairs** of the span-level relation score.
3. **Decision**: B0/Z0 predict every (h, t, r) with `S[h, t, r] ≥ rt`.

**Z0** — zero-shot. Entity labels: 23 fine-grained descriptors mapped to the 6 types (`relex_io.V3_TYPES`, label set "v3"); relations: the 96 names. Configuration: entity threshold 0.3, `flat_ner=True`, `adjacency_threshold=0.5`, relation threshold `rt = 0.35`.

**B0** — fine-tuned. Training examples (`relex_io.to_train_relex`): one NER span per gold mention (6 labels: `person, organization, location, time, number, miscellaneous`) and, for each gold triple, a relation between **every pair** of head/tail mentions. Hyperparameters (identical for all runs of B0): learning rate 2e-5 (constant, no warm-up), bf16, seed 42, batch size 5 (N=10) / 10 (N=50), training from the pretrained checkpoint on one H100 per (N, subset). Checkpoints are saved at fixed steps (N=10: 20, 40, 80, 160, 320, 480; N=50: 25, 50, 100, 200, 400, 800); the checkpoint, the entity threshold ∈ {0.3, 0.5} and the relation threshold `rt` ∈ {0.10, 0.15, …, 0.95} are selected per subset on the selection set D (see §3) by maximizing the main metric (ties: earlier checkpoint, then lower thresholds). Selected values are in `results/test/baseline_metrics.csv` (`rt`, `ent_thr`) and `results/selection/ftB_baseline_dev_D50.json`.

**J0** — B0 + judge (`relex_router.py`, `llm_router.py`, `run_judge.py`). For each document and each ordered entity pair, candidate relations are those with `S ≥ lo` whose (head type, tail type) is allowed by the domain/range table of Menotti et al. (`DocRE-StructuralConsistency`):

* `S ≥ hi` → accepted directly;
* `lo ≤ S < hi` → sent to the judge (at most 5 options per pair, ordered by score);
* `S < lo` → discarded.

The judge (`openai/gpt-oss-120b` on DeepInfra, `reasoning_effort=medium`, temperature 0, seed 0, JSON output) receives the document with sentence ids, the entities involved, **up to 8 solved examples taken from the N training documents** (true and rejected candidates) and up to 30 questions per call; for each question it selects any subset of the options or none (system prompt `SYSTEM_V3` in `llm_router.py`). Fusion is *hard*: in routed pairs the prediction is exactly the judge's choice; if a response cannot be parsed (or the call fails), the extractor's own decision (`S ≥ rt`) is kept for those options. Bands, chosen on D with the real judge: **N=10 → [0.1, 0.7)**, **N=50 → [0.05, 0.9)**. The extractor is exactly B0 (same checkpoints and thresholds), so J0 − B0 isolates the effect of the judge.

## 3. Evaluation Methodology

**Data and partitions**

* Training: `train_N{10,50}_s{0..4}` (5 subsets per N).
* **Selection set D**: 50 dev documents, `numpy.random.default_rng(1234).permutation(500)[:50]` (sorted). **Every** choice (thresholds, checkpoint, label set, judge band) was made on D only; the same D is used for all pipelines.
* **Test**: the 500 test documents; each frozen configuration was evaluated on test; no choice was made on test.
* Equal supervision: Z0, B0 and J0 use at most the N gold training documents plus D for selection.
* Mention-level control of LLM memory: the judge was re-run on the first 100 test documents with PER/ORG/LOC/MISC mentions replaced by `ENT<i>` in the prompt (TIME/NUM kept), for N=10 (5 subsets) and N=50 (subsets 0, 1).

**Protocol.** For each pipeline: produce raw model outputs (Modal), build predicted documents and S offline, apply the frozen configuration, score against gold. Statistics: mean ± standard deviation over the 5 subsets; **paired bootstrap over documents** (10 000 resamples, seed 0; the same resampled documents are used in the 5 subsets and the statistic is the mean over subsets of the F1 difference). Subsets share dev/test, so the intervals reflect document sampling, not training-subset variability.

**Comparison criteria.** Pipelines are compared on identical test documents, identical extractor checkpoints (J0 vs B0), identical judge server/model/prompt, and the main metric below. Cross-model or cross-protocol comparisons with published numbers are not made.

**Considerations needed to interpret the results**

* Dev-selected values are optimistic: B0 with N=10 averages 25.9 F1 on D (`dev_f1` column of `baseline_metrics.csv`) vs 23.6 on test. Only test numbers are reported as results.
* Judge responses are cached by prompt hash. The J0 test numbers were obtained with the response cache of the full set of calls; calls that had failed with API rate limiting (HTTP 429) in a previous partial pass of the same configuration were re-issued. The minimum valid-response rate over the ten J0 runs is 98.8 % (`valid_rate` in `results/test/J0_rows.csv`).
* The judge uses the Menotti domain/range table only as a candidate filter, and its few-shot examples come from the N training documents.
* `adjacency_threshold` has no effect on the output in `gliner==0.2.29` in the zero-shot grid (identical scores for 0.3, 0.5, 0.65 in `results/selection/zs_dev_D50.csv`); it is fixed at 0.5.

## 4. Metrics

All triple metrics are **micro-averaged over the test split** and reported in %.

* **Main metric — end-to-end strict typed triple F1** (`eval_e2e.re_e2e(mode='strict', typed=True)`). A predicted entity matches a gold entity iff their sets of mentions `{(sent_id, start, end)}` are identical **and** the entity types are equal (one-to-one; duplicates matched in order). A predicted triple (h, r, t) is correct iff both entities match gold entities and (h_gold, r, t_gold) is a gold triple; duplicates after matching count once; every other prediction is a false positive; the recall denominator is **all** gold triples (including those whose entities were not predicted).
  P = TP / #pred, R = TP / #gold, F1 = 2PR / (P + R).
* **Strict untyped F1**: same without the type condition. **Soft typed F1**: entity matching by Jaccard of mention sets ≥ 0.5 with Hungarian one-to-one assignment, types required.
* **Ign F1** (`eval_e2e.ign_e2e`): as in the official Re-DocRED script, correct triples whose (head mention name, tail mention name, relation) appears in the **full train split** (`train_revised.json`, 3 053 documents) are removed from the precision: `P_ign = (TP − TP_train) / (#pred − TP_train + 1e-5)`; recall unchanged; `Ign F1 = 2·P_ign·R / (P_ign + R)`.
* **Entity metrics** (`entity_metrics`): typed mention F1 (`mention_typed_f1`), strict cluster F1 (`cluster_strict_f1`) and type accuracy of matched entities (`type_acc`).
* **Verification against the official evaluator.** With gold entities, `eval.evaluate` reproduces `official_evaluate` of the Re-DocRED repository (`evaluation.py`) for F1, P, R and Ign F1; `tests/test_eval_official.py` checks 7 cases (gold, subsets, noisy, random, duplicated, empty; with/without train facts) with tolerance 1e-9, and `tests/test_eval_e2e_ign.py` checks that `ign_e2e` equals `eval.evaluate` when entities are gold. Both were executed (all cases passed) during development; their outputs are not stored in this repository and they were not re-run when it was prepared.

Why these metrics: micro F1 and Ign F1 are the standard DocRE measures (Yao et al., 2019; Tan et al., 2022); an end-to-end, typed matching is required because the extractor predicts its own entities, and a typed strict match avoids crediting relations attached to entities of the wrong type or with wrong boundaries (the soft variant is reported to show the effect of boundary/grouping errors).

## 5. Experimental Environment

| Component | Value |
|---|---|
| Local analysis / judge client | Python 3.12.3; numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, openai 2.41.1, modal 1.6.2, scikit-learn 1.8.0 (needed only by the official Re-DocRED evaluator in the tests) (`requirements.txt`); 4 CPU cores, ≈7.6 GiB RAM, no GPU. `dateparser` was **not** installed in this environment |
| Modal container | Python 3.11 (`debian_slim`); `gliner==0.2.29`; torch 2.14.1+cu130 (unpinned, version recorded in run output); `numpy<2.3`, `accelerate>=1.1.0`, `scipy`, `huggingface_hub` (unpinned); `dateparser==1.4.3`; `transformers` as required by `gliner` (<5.17.0; exact version not recorded) |
| GPUs (Modal) | NVIDIA A100-SXM4-40GB (inference), NVIDIA H100 80GB HBM3 (fine-tuning) |
| Model | `knowledgator/gliner-relex-large-v1.0` (Hugging Face; revision hash not recorded) |
| Judge | DeepInfra OpenAI-compatible API, `openai/gpt-oss-120b`, `reasoning_effort=medium`, temperature 0, seed 0, `response_format=json_object`; per-call wall limit 300 s; up to 12 back-off retries on HTTP 429; up to 200 simultaneous calls per process (`C1_DI_CONCURRENCY`) |
| Data | Re-DocRED commit `ccfb54f5ddf5836027c87badda10f6dfc56efaac`; DocRE-StructuralConsistency commit `ee9de0f8cf9f2df3b061d72e13568d6361cc8dd0` (`data/download.sh`) |
| Dates | all runs: 2026-10-20 / 2026-10-21 |

Environment note: entity grouping (`ent_cluster.time_key`) canonicalizes dates with `dateparser` when it is importable and falls back to the normalized name otherwise. Checkpoint/threshold selection for B0 ran in the Modal container (with `dateparser`); every offline evaluation reported here (Z0 selection, band selection, all test metrics, J0) ran in the local environment (without it). On the 50 selection documents the offline recomputation reproduced the container's selection scores for all 10 B0 models; the effect on the test set was not measured.

## 6. Final Results

Test split, 500 documents; F1 = end-to-end strict typed triple F1 (%). Mean ± standard deviation over the 5 training subsets. Source: `results/test/*.csv`.

| Pipeline | N | F1 | Ign F1 | Precision | Recall | F1 untyped | F1 soft typed |
|---|---|---|---|---|---|---|---|
| Z0 (zero-shot) | – | 16.84 | 15.63 | 18.60 | 15.38 | 18.29 | 27.33 |
| B0 | 10 | 23.62 ± 2.01 | 22.36 ± 1.88 | 25.93 ± 3.03 | 21.75 ± 1.43 | 25.16 ± 2.00 | 33.94 ± 3.10 |
| **J0** | 10 | **25.85 ± 1.62** | 24.42 ± 1.42 | 26.00 ± 2.86 | 26.18 ± 3.34 | 27.56 ± 1.71 | 36.99 ± 2.63 |
| B0 | 50 | 31.34 ± 0.71 | 29.87 ± 0.44 | 33.74 ± 2.16 | 29.50 ± 2.48 | 32.85 ± 1.19 | 42.97 ± 0.96 |
| **J0** | 50 | **33.41 ± 0.15** | 31.68 ± 0.32 | 31.82 ± 2.10 | 35.45 ± 2.46 | 35.04 ± 0.41 | 45.48 ± 0.21 |

**Judge effect (J0 − B0)**, paired bootstrap over documents (`A1_A2_bootstrap.csv`):

| N | ΔF1 | 95 % CI | Δ Ign F1 | Subsets with J0 > B0 |
|---|---|---|---|---|
| 10 | +2.23 | [1.88, 2.60] | +2.06 | 5 / 5 |
| 50 | +2.07 | [1.64, 2.52] | +1.81 | 5 / 5 |

(no resample out of 10 000 had a non-positive difference). The gain comes from recall (+4.4 / +6.0); precision is unchanged for N=10 and 1.9 points lower for N=50.

**Per subset (F1, test)** — from `results/test/J0_rows.csv`:

| N | s0 | s1 | s2 | s3 | s4 |
|---|---|---|---|---|---|
| 10: B0 → J0 | 24.44 → 27.26 | 20.38 → 23.59 | 23.42 → 24.82 | 25.79 → 27.34 | 24.07 → 26.26 |
| 50: B0 → J0 | 31.55 → 33.18 | 31.56 → 33.36 | 31.60 → 33.50 | 30.10 → 33.55 | 31.91 → 33.49 |

**LLM-memory control** (first 100 test documents; ΔF1 = judge − B0; `results/masking_control/judge_mask.csv`):

| N / subset | 10/s0 | 10/s1 | 10/s2 | 10/s3 | 10/s4 | 50/s0 | 50/s1 |
|---|---|---|---|---|---|---|---|
| names visible | +2.61 | +2.86 | +0.79 | +0.27 | +1.39 | +0.48 | +1.09 |
| names masked | +0.41 | +1.37 | +0.66 | +0.57 | +1.66 | +1.22 | −0.74 |

With 100 documents the estimates are noisy; the judge's gain is reduced but not eliminated when entity names are hidden for N=10 (mean of the five rows: +1.58 visible vs +0.93 masked).

**Selection evidence** (set D, `results/selection/`): `zs_dev_D50.csv` and `zs_labels_D50.csv` (zero-shot grid and label sets: best strict typed F1 on D — v0 6 plain names 3.5, v1 with descriptions 14.4, v3 23 descriptors 14.9; typed mention F1 on D at the best configuration: 44.0 with v0 vs 77.0 with v3), `ftB_baseline_dev_D50.json` (B0 checkpoint/threshold tables), `judge_band_selection_D50.csv` (judge runs for the band selection; `src/select_band.py`).

## 7. Technical Decisions

* **Strict, typed, end-to-end matching** is the main metric; soft matching is reported as a secondary view of boundary/grouping errors.
* **Relation score at entity level = max over mention pairs**; **training relations are expanded to all mention pairs** of each gold triple, consistent with that aggregation.
* **Label sets**: zero-shot uses 23 fine-grained descriptors mapped to the 6 types (v3), chosen on D; fine-tuned models use the 6 plain names (the model learns them from data).
* **Explicit judge band and top-k** (no multiplicative band rule); the number of candidates cut by top-k is logged (`trunc` in `J0_rows.csv`: 16 of 32 657 candidates for N=10/s4, 45 of 36 320 for N=50/s0, 2 of 22 913 for N=50/s4, 0 elsewhere). The type filter (Menotti table) is applied before routing.
* **Fallback on invalid judge output**: the extractor's decision is kept (no pair is dropped because of a parsing failure).
* **Single judge server and prompt** for all comparisons; temperature 0; few-shot examples only from the N training documents.
* **Selection only on D** (50 dev documents, same for all pipelines); ties resolved toward the earlier checkpoint and lower thresholds; test is guarded in the code (`ALLOW_TEST=1`).
* **Limit of the Z0 threshold evidence**: the Z0 relation threshold 0.35 was chosen on D from a finer grid (steps of 0.05) than the one stored in `zs_labels_D50.csv` (0.3, 0.4, …); the finer table is not included, so that single value is not reproducible from the stored tables (`dev_f1 = 15.0` for Z0 in `baseline_metrics.csv` is a constant recorded from that selection).
* **Edge of the grid**: for N=50/s3 the selected relation threshold is the largest value of the grid (0.95).

## 8. Repository Structure

```
README.md
requirements.txt             local environment (versions verified)
data/
  download.sh                clones Re-DocRED and DocRE-StructuralConsistency at the evaluated commits
  subsets/redocred/          train_N{10,50}_s{0..4}.json — the 10 low-resource training sets
modal/relex_modal.py         Modal functions: inference, fine-tuning, checkpoint/threshold selection on D (modes listed in its docstring)
src/
  data.py                    paths, Re-DocRED loading, 96-relation maps; RUNS = output dir of new executions (never results/)
  relex_io.py                label sets (v0, v1, v3), document → text, token/char offsets, training examples
  relex_post.py              raw model output → predicted document and S[h,t,r]
  ent_cluster.py, ent_build.py   rule-based mention → entity grouping and predicted-document construction
  constraints.py             domain/range type table (Menotti et al.) used as candidate filter
  eval.py                    gold-entity metrics (reimplementation of the official script)
  eval_e2e.py                end-to-end metrics: strict/soft matching, typed, Ign F1, entity metrics, per-document counts
  ft_select.py               selection sweep on D (entity × relation thresholds), tie-break rule
  runs.py                    loads B0 raw outputs into (gold, predicted docs, S, selected config)
  relex_router.py            judge band (sure / routed / discarded), hard fusion with fallback, entity masking
  llm_router.py              judge: prompt, few-shot examples, DeepInfra client, cache, rate-limit handling, spend cap
  run_judge.py               judge runs for band selection (dev), test evaluation, and the masking control
  collect_judge_rows.py      run_judge JSON lines → CSV
  select_band.py             band choice rule on D
  eval_zs_dev.py, eval_zs_labels.py   zero-shot selection tables on D
  eval_test.py               test metrics of B0 and Z0
  analyze_test.py            J0 vs B0 summary tables and paired bootstrap
tests/                       equivalence tests of the metrics (official script; Ign F1)
results/                     OFFICIAL RESULTS (read-only; code never writes here)
  test/                      baseline_metrics.csv (Z0, B0), J0_rows.csv, J0_counts_*.npz and baseline_counts.npz (per-document counts),
                             A1_A2_summary.csv, A1_A2_bootstrap.csv
  selection/                 selection tables on D (see §6)
  masking_control/judge_mask.csv
```

## 9. Usage

**Consult the official results** (no execution needed): `results/test/` (final numbers), `results/selection/` (how configurations were chosen), `results/masking_control/`.

**Recompute the summary tables from the included per-document counts** (no new experiment; does not call any model or API):

```bash
pip install -r requirements.txt
mkdir -p runs && cp -r results/test runs/test        # code writes to runs/, never to results/
python src/analyze_test.py                            # regenerates A1_A2_summary.csv and A1_A2_bootstrap.csv in runs/test
```

**Check the metric implementation against the official evaluator** (needs the datasets):

```bash
bash data/download.sh
python tests/test_eval_official.py
python tests/test_eval_e2e_ign.py
```

**Re-run the evaluation** (new executions: requires a Modal account with GPU credit and a DeepInfra key in `.env`; results are written to `runs/`, or to `$RELEX_RUNS_DIR`). Order of the stages:

1. `bash data/download.sh`.
2. Zero-shot selection: `modal run modal/relex_modal.py --mode zs_dev` and `--mode zs_labels`, then `python src/eval_zs_dev.py`, `python src/eval_zs_labels.py`.
3. B0: `--mode train_b0` (writes `runs/selection/ftB_baseline_dev_D50.json`), then `--mode infer_dev`.
4. Judge band selection on D: `python src/run_judge.py --split dev --n N --sub K --lo LO --hi HI` over the band grid {(0.05,0.9), (0.1,0.9), (0.2,0.9), (0.1,0.7), (0.2,0.7)}, subsets 0 and 1, N ∈ {10, 50}; then `python src/collect_judge_rows.py runs/selection/judge_runs_dev.jsonl runs/selection/judge_band_selection_D50.csv` and `python src/select_band.py runs/selection/judge_band_selection_D50.csv`.
5. Test (frozen configurations only; `ALLOW_TEST=1`): `--mode infer_test`, `--mode zs_test`, `python src/eval_test.py`, then `python src/run_judge.py --split test --n N --sub K --lo LO --hi HI` for the 10 (N, subset) pairs with the selected bands, `collect_judge_rows.py` on `runs/test/judge_runs_test.jsonl` → `runs/test/J0_rows.csv`, and `python src/analyze_test.py`.
6. Masking control: `python src/run_judge.py --split test --first 100 [--mask-entities] ...`.

Judge environment: `set -a; source .env; set +a` (keys `DEEPINFRA_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`), `C1_MAX_USD` (spend cap), `C1_DI_CONCURRENCY` (200 in the reported test runs). Raw model outputs (≈1.1 GB), the judge response cache and the trained checkpoints are not included in this repository; the judge's responses come from a live API and can differ if the cache is not available.
