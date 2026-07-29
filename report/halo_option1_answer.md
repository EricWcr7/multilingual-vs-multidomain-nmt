# Multilingual vs. Multi-domain NMT: A Controlled Experimental Plan with Multilingual Grounding Diagnostics

**HALO undergraduate challenge — Option 1**

**Applicant:** Eric Wang

**Date:** July 29, 2026

## 1. Executive summary and interpretation

This submission answers the “Multilingual vs. Multi-domain NMT experiment plan” challenge in two deliberately separate parts:

1. a **proposed** equal-budget LoRA experiment comparing language diversity, domain diversity, and a matched Mandarin-news control; and
2. an **executed** zero-shot demonstration that validates the data, inference, evaluation, and reporting pipeline on English→Gujarati, Georgian, Tamil, and Simplified Chinese.

The research question is:

> Under an equal adaptation budget, how does allocating data across target languages versus across domains affect translation quality, robustness, and faithfulness to the English source?

English is fixed as the source. Gujarati, Georgian, and Tamil preserve continuity with the crossed historical workflow linked from the challenge. Simplified Chinese is added as the anchor target because it supports both the multilingual and multi-domain comparisons and permits a bilingual audit.

The instrument is `facebook/nllb-200-distilled-600M`. NLLB is not presented as 2026 state of the art; it is a Colab-friendly checkpoint that supports all four targets. The demonstration contains 190 real translations and no LoRA training. Its metrics, 20-output Mandarin audit, and ten counterfactual pairs demonstrate the measurement pipeline; they do not compare adaptation strategies.

**Evidence boundary.** Data validation, deterministic sampling, zero-shot inference, automatic diagnostics, the Mandarin audit, and counterfactual checks were executed. The three LoRA arms and three-seed analysis are proposed only. COMET was omitted under the deadline safeguard after the core artifacts succeeded.

**Reproducibility:** [GitHub repository](https://github.com/EricWcr7/multilingual-vs-multidomain-nmt/tree/codex/halo-modern-pilot) · [Executable Colab](https://colab.research.google.com/github/EricWcr7/multilingual-vs-multidomain-nmt/blob/codex/halo-modern-pilot/notebooks/halo_option1_demo.ipynb)

**Pinned model revision:** `f8d333a098d19b4fd9a8b18f94170487ad3f821d`

**Final recorded artifacts:** 2026-07-29 19:36:19 UTC (15:36 Toronto), Python 3.13.5, PyTorch 2.10.0, Transformers 4.53.0, Hugging Face Hub 0.33.1, SacreBLEU 2.5.1, CPU. This was a fresh inference run from a snapshot whose seven required files matched the pinned hashes; no prediction cache was used. A separate full-sample verification run completed all 13 code cells in order on a Colab Tesla T4 on July 29, 2026; it wrote all five expected result artifacts and the downloadable ZIP without a Python traceback or failed assertion. Both runs used the same inputs, generation settings, and verified seven-file model fingerprint, but their predictions and derived metrics were not byte-identical across CPU and CUDA; `examples.md` was identical. The committed tables and every number below come only from the recorded CPU run.

## 2. Research question and preregistered hypotheses

The full experiment changes the adaptation-data allocation while fixing the checkpoint, English source, total examples, optimization, sequence lengths, decoding, and checkpoint rule.

1. **Cross-language transfer:** multilingual adaptation will most clearly improve Gujarati, Georgian, and Tamil news relative to the Mandarin-news control because those targets receive direct supervision.
2. **Cross-domain robustness:** multi-domain Mandarin adaptation will most clearly improve Mandarin social, speech, and literary cells relative to the control.
3. **Matched-condition specialization:** the Mandarin-news control may perform best on Mandarin news because all auxiliary data reinforce that cell.
4. **Quality is not faithfulness:** higher chrF++ or BLEU need not imply better digit preservation, fewer unsupported additions, or better counterfactual sensitivity.
5. **No context-free winner:** each strategy targets different cells; a single aggregate ranking would mislead.

These are hypotheses for the LoRA study, not conclusions from the one-system demonstration.

## 3. Historical continuity and Mandarin

The crossed workflow is provenance, not a current recipe. I retain its English→Gujarati, English→Georgian, and English→Tamil directions while using pinned direct model loading, current datasets, document-aware splits, licenses, hashes, and grounding diagnostics.

Mandarin (`zho_Hans`) supplies the common target for both diversity axes. WMT24++ provides four Mandarin domains [4], and I can read English and Mandarin for direct faithfulness review. Mandarin is not a proxy for other languages, so results remain language–domain specific.

## 4. Model and data

### Model

The notebook directly loads NLLB with `AutoTokenizer` and `AutoModelForSeq2SeqLM`; the current model card marks the Transformers v5 translation pipeline unsupported [1]. Before loading, it verifies the SHA-256 value of every required model, configuration, and tokenizer file against the pinned revision; a local override is accepted only when all seven files are byte-identical. The inference-cache identity binds those verified hashes, the generation configuration, and hashes of every selected input. It asserts `eng_Latn`, `guj_Gujr`, `kat_Geor`, `tam_Taml`, `zho_Hans`, and forced target BOS tokens, then uses beam size 4. The CC BY-NC 4.0 checkpoint is used only for non-commercial research. `m2m100_418M` is an unmixed fallback [3].

### Data

| Resource | Use | Controls and caveats |
|---|---|---|
| NTREX-128 [5] | Comparable news condition for all four targets | 1,997 aligned segments, 123 document IDs; CC BY-SA 4.0 |
| WMT24++ [4] | Mandarin news, social, speech, literary cells | Canary and all `is_bad_source=true` rows removed; Apache 2.0; literary has only 8 documents |
| TICO-19 [6,7] | Adaptation-independent Mandarin medical test | Official test only; never used for allocation or tuning |
| FLORES+ [8] | Possible later evaluation | Excluded from the required artifact because access is gated |

These are **custom teaching splits**, not official benchmark submissions. I do not claim that NTREX or TICO material was absent from NLLB pretraining.

Each normalized row has:

`example_id, document_id, source, reference, target_language, domain, dataset, split`

The pipeline uses stable 70/10/20 document splits, normalized source hashes, and aligned NTREX document assignments. It removes empty/duplicate rows and rejects inputs or references over 256 model tokens instead of truncating. Assertions confirm zero canary or bad-source rows, within-corpus document/hash separation, zero source-hash overlap between every proposed arm and the complete cross-corpus evaluation pool, fixed unique IDs, valid language tokens, and nonempty outputs.

After filtering, the limiting training cell is WMT24++ Mandarin speech with 77 examples, so the frozen budget is **\(B=77\)**. Each adapted arm therefore contains **\(4B=308\)** examples.

## 5. Equal-budget experimental matrix

| System | Proposed adaptation data | Examples |
|---|---|---:|
| Unadapted baseline | Original NLLB checkpoint | 0 |
| Mandarin-news control | Shared \(B\) Mandarin-news + \(3B\) other Mandarin-news | 308 |
| Multilingual | \(B\) each Mandarin, Gujarati, Georgian, Tamil news | 308 |
| Multi-domain | \(B\) each Mandarin news, social, speech, literary | 308 |

The identical \(B\) Mandarin-news examples occur in every adapted arm; auxiliary samples are non-overlapping within each arm. The control distinguishes diversity effects from simply receiving more adaptation.

The proposed LoRA configuration is `r=8`, alpha 16, dropout 0.05 on `q_proj`/`v_proj`; learning rate `2e-4`; batch 4; gradient accumulation 4; 3 epochs; 5% warmup; maximum lengths 256; fixed-length padding; seeds 13, 42, and 73; and final-checkpoint comparison. Each arm has 60 optimizer steps per seed and 157,696 padded tokens per epoch. Non-padding tokens differ: control 20,635; multilingual 23,954; multi-domain 36,910. Runtime and peak GPU memory will be logged. Domain tags and ITTL remain future ablations.

## 6. Evaluation and statistical plan

**chrF++** is the primary broadly comparable automatic metric [9]. **SacreBLEU** is reported per cell, never averaged across scripts [10]. Gujarati, Georgian, and Tamil use tokenizer `intl`; Chinese uses `zh`. Full signatures are:

- `nrefs:1|case:mixed|eff:yes|tok:intl|smooth:exp|version:2.5.1`
- `nrefs:1|case:mixed|eff:yes|tok:zh|smooth:exp|version:2.5.1`

COMET (`Unbabel/wmt22-comet-da`) is secondary because learned-metric validity can vary across low-resource languages and domains [11,12]. It was not required for the deadline-critical demonstration.

The automatic faithfulness diagnostic extracts normalized digit-bearing facts and simple date/month variants after NFKC normalization. It reports source-fact recall and unsupported predicted-digit rate. It does **not** understand units, written-out numbers, or the surrounding proposition; those extensions belong in the full study.

The audit labels 10 Mandarin news and 10 medical outputs across seven error categories and four severities. A counterfactual pair is eligible to pass only when the original output contains the old fact and not the replacement; the changed output must then contain the replacement, omit the old fact, and preserve unrelated meaning (masked chrF++ ≥70). Numeric markers use digit-boundary-aware matching.

The full LoRA analysis uses paired document-level bootstrap resampling (1,000 replicates), reports each cell and seed separately, and makes no aggregate-winner claim. The full Mandarin audit adds a second bilingual annotator, blind system labels, agreement reporting, and adjudication.

## 7. Executed demonstration results

The notebook executed from the first cell through artifact generation with 13/13 code cells successful. It produced 190 nonempty predictions: 100 NTREX, 45 WMT24++, 25 TICO, and 20 counterfactual translations. Fresh CPU inference took 434.21 seconds after verifying all seven pinned model/tokenizer files; the new input-and-model-bound cache was not used. The generated predictions and metric table are byte-identical to the earlier run, while the manifest now records the enforced provenance and zero global arm-to-evaluation source-hash overlaps. NTREX rejected 0 overlength rows; WMT24++ and TICO rejected 1 each. All 20 audit rows are reviewed. COMET and LoRA were not run.

| Dataset / cell | n | chrF++ | BLEU | Digit recall | Spurious digits |
|---|---:|---:|---:|---:|---:|
| NTREX Gujarati news | 25 | 48.01 | 18.10 | 14/17 (82.4%) | 2/16 (12.5%) |
| NTREX Georgian news | 25 | 47.07 | 18.24 | 13/17 (76.5%) | 3/16 (18.8%) |
| NTREX Tamil news | 25 | 45.97 | 11.64 | 14/17 (82.4%) | 1/15 (6.7%) |
| NTREX Mandarin news | 25 | 22.77 | 29.69 | 12/17 (70.6%) | 2/14 (14.3%) |
| TICO Mandarin medical | 25 | 34.62 | 40.51 | 37/41 (90.2%) | 3/40 (7.5%) |
| WMT Mandarin social | 15 | 28.16 | 22.82 | 2/6 (33.3%) | 0/2 (0.0%) |
| WMT Mandarin speech | 15 | 17.46 | 23.45 | 12/15 (80.0%) | 2/14 (14.3%) |
| WMT Mandarin literary | 15 | 12.13 | 13.15 | 1/8 (12.5%) | 2/3 (66.7%) |

These values compare zero-shot test conditions, not adaptation strategies. In particular, BLEU is tokenized differently across script groups, WMT24++ domains have different reference conventions, and the digit denominators are small. The complete metric signatures and counts are saved in `demo_metrics.csv`.

Public files contain IDs, predictions, scores, and metadata—not bulk corpus text or references. Their exact SHA-256 checksums are recorded and verified in `demo_manifest.json`.

## 8. Mandarin audit and sensitivity

The 20-output single-annotator audit found: unsupported addition 0, omission 8, contradiction 6, entity/number mutation 5, non-translation 0, repetition 0, and fluency-only error 1. Severity was 7 none, 4 minor, 9 major, and 0 critical. Labels overlap, so category totals do not equal 20.

Examples show why overlap is insufficient. `ntrex:1655:zho_Hans` mistranslates “backbencher” and drops the central claim; `ntrex:0044:zho_Hans` loses or changes “shark,” lobster diving, and Saturday. By contrast, `ntrex:1970:zho_Hans` preserves its main content. TICO `PubMed_8:599` corrupts two animal entities, showing that high medical BLEU does not guarantee entity faithfulness.

Six of ten counterfactual pairs met all criteria. Year, calendar-date, decision polarity, directional polarity, weekday, and patient-count edits passed. City and organization edits changed the named fact but destabilized unrelated wording; the 12%→7% edit failed the prerequisite because the original output had already omitted 12 and its unrelated wording also changed; and the \$2.1M edit failed to render the new amount correctly. This is evidence of mixed source sensitivity, not a system-wide rate.

## 9. Expected full-study outcomes — hypotheses, not findings

The plausible result is specialization: multilingual adaptation helps directly supervised target languages on news; multi-domain adaptation helps Mandarin domain shifts; and the control performs well on Mandarin news. Negative transfer is also plausible in a 600M multilingual model. An informative result may be disagreement between overlap and grounding diagnostics—for example, higher chrF++ with more entity mutations.

Any conclusion will therefore be conditional:

> Under this checkpoint, datasets, and budget, allocation strategy A improved outcome X on cells Y while changing faithfulness diagnostic Z.

It will not claim that multilingual training is universally better than multi-domain training, or the reverse.

## 10. Limitations, HALO connection, and next steps

Language and domain remain confounded with corpus, reference style, script, supervision, and document count. Pretraining exposure is unknown; equal examples do not equal useful-token compute; literary has eight documents; the digit heuristic is narrow; and the audit has one annotator and 20 outputs. One unadapted checkpoint supports no significance or mitigation claim.

The HALO connection is methodological and bounded:

> Translation is a controlled conditional-generation setting in which the source is the grounding context. This study tests sensitivity and faithfulness to that context across languages and domains; it does not establish that translation hallucinations generalize to open-world LLM hallucinations or evaluate a HALO mitigation method.

The next study should execute all three LoRA arms for all seeds, add document-bootstrap intervals, complete a two-annotator Mandarin audit, evaluate FLORES+ after accepting its terms, and only then ablate domain tags, ITTL, adapter composition, or sampling policy. A separate experiment could use the strongest observed failure mode to test multilingual grounding regularization, cross-lingual calibration, or sensitivity-based filtering.

## 11. Reproducibility

The repository contains the executed notebook, Markdown report, pinned requirements, manifest, predictions, cell-level metrics, audit labels, and selected examples. Raw datasets and checkpoints are downloaded into temporary storage and are not committed. The manifest freezes sample IDs, splits, global arm/evaluation leakage checks, arm allocations, versions, all required model/tokenizer hashes, the input/cache fingerprints, language-token IDs, generation settings, runtime, and the fact that LoRA and COMET were not executed. Every number in this report comes from the saved artifacts.

## References

1. Meta AI. “NLLB-200 Distilled 600M” model card, accessed July 2026. <https://huggingface.co/facebook/nllb-200-distilled-600M>
2. NLLB Team et al. “No Language Left Behind: Scaling Human-Centered Machine Translation.” *arXiv:2207.04672*, 2022. <https://arxiv.org/abs/2207.04672>
3. Fan, A. et al. “Beyond English-Centric Multilingual Machine Translation.” *JMLR* 22, 2021. <https://jmlr.org/papers/v22/20-1307.html>
4. Deutsch, D. et al. “WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects.” *arXiv:2502.12404*, 2025. <https://arxiv.org/abs/2502.12404>
5. Federmann, C., Kocmi, T., and Xin, Y. “NTREX-128 — News Test References for MT Evaluation of 128 Languages.” *SUMEval 2022*. <https://aclanthology.org/2022.sumeval-1.4/>
6. Anastasopoulos, A. et al. “TICO-19: The Translation Initiative for COvid-19.” *NLP-COVID19 at EMNLP 2020*. <https://aclanthology.org/2020.nlpcovid19-2.5/>
7. TICO-19 project. Official dataset. <https://tico-19.github.io/>
8. Goyal, N. et al. “The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation.” *TACL* 10, 2022; FLORES+ portal. <https://huggingface.co/datasets/openlanguagedata/flores_plus>
9. Popović, M. “chrF: Character n-gram F-score for Automatic MT Evaluation.” *WMT 2015*. <https://aclanthology.org/W15-3049/>
10. Post, M. “A Call for Clarity in Reporting BLEU Scores.” *WMT 2018*. <https://aclanthology.org/W18-6319/>
11. Rei, R. et al. “COMET: A Neural Framework for MT Evaluation.” *EMNLP 2020*. <https://aclanthology.org/2020.emnlp-main.213/>
12. Unbabel. “wmt22-comet-da” model card. <https://huggingface.co/Unbabel/wmt22-comet-da>
13. Hu, E. J. et al. “LoRA: Low-Rank Adaptation of Large Language Models.” *ICLR 2022*. <https://openreview.net/forum?id=nZeVKeeFYf9>
14. Guerreiro, N. M. et al. “Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation.” *EACL 2023*. <https://aclanthology.org/2023.eacl-main.75/>
