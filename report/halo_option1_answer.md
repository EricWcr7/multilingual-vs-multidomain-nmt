# Multilingual vs. Multi-domain NMT: Experimental Plan

**HALO undergraduate challenge — Option 1**

**Applicant:** Yi Fan (Eric) Wang

**Date:** July 29, 2026

## 1. My interpretation of the challenge

My research question is:

> Under the same adaptation budget, how does allocating data across target languages rather than across domains affect translation quality, robustness, and faithfulness to the English source?

I would keep English as the source language and translate into Gujarati, Georgian, Tamil, and Simplified Chinese. Gujarati, Georgian, and Tamil preserve continuity with the historical workflow linked from the challenge page. I add Simplified Chinese as an anchor language because it can be used in both sides of the comparison and because I can conduct a direct English–Mandarin error analysis.

My goal would not be to declare multilingual or multi-domain training universally better. Instead, I would identify where each allocation strategy helps, where it hurts, and whether improvements in standard translation metrics are accompanied by better reliance on the source.

## 2. Proposed experiment

### 2.1 Systems to compare

I would start every system from the same `facebook/nllb-200-distilled-600M` checkpoint. NLLB is not presented as the best translation model available in 2026. I selected it because a single Colab-friendly checkpoint supports all four required target languages, which makes a controlled comparison possible [1,2].

After filtering and document-level splitting, the smallest eligible training cell contains 77 examples. I therefore freeze the shared budget at **\(B=77\)** and give every adapted system **\(4B=308\)** examples.

| System | Adaptation data | Purpose |
|---|---|---|
| Unadapted baseline | Original NLLB checkpoint | Measures performance before task-specific adaptation |
| Mandarin-news control | Shared \(B\) Mandarin-news examples plus \(3B\) additional Mandarin-news examples | Controls for receiving more data without adding language or domain diversity |
| Multilingual | \(B\) each of Mandarin, Gujarati, Georgian, and Tamil news | Tests allocation across target languages while holding the news domain fixed |
| Multi-domain | \(B\) each of Mandarin news, social, speech, and literary text | Tests allocation across domains while holding the target language fixed |

The same \(B\) Mandarin-news examples would appear in all three adapted systems. The remaining examples would be non-overlapping within each system. This shared anchor is important: without the Mandarin-news control, an improvement could come from receiving more examples rather than from multilingual or multi-domain diversity.

### 2.2 Data

I would use three resources:

| Resource | Role in the plan | Important limitation |
|---|---|---|
| NTREX-128 [4] | The comparable news condition for all four target languages | It is a relatively small news test collection |
| WMT24++ [3] | Mandarin news, social, speech, and literary conditions | The domains differ in corpus and reference style; literary contains only eight documents |
| TICO-19 [5,6] | A Mandarin medical test that remains independent of adaptation | It tests a new domain but does not prove absence from model pretraining |

These would be custom splits for this exercise, not official benchmark submissions. I would use stable 70/10/20 document-level splits, assign aligned NTREX documents identically across languages, and hash normalized English sources to prevent cross-split overlap. Empty rows, duplicates, WMT24++ canary rows, rows marked `is_bad_source=true`, and examples longer than 256 source or target tokens would be rejected rather than silently truncated.

FLORES+ would be useful as a later evaluation set, but I would keep it outside the required submission because it is gated and requires accepting its terms [7].

### 2.3 Adaptation procedure

To keep the experiment feasible, I would use LoRA rather than fully fine-tuning all 600 million parameters [12]. The proposed configuration is:

- LoRA rank 8, alpha 16, dropout 0.05 on `q_proj` and `v_proj`;
- learning rate `2e-4`, batch size 4, and gradient accumulation 4;
- three epochs with 5% warmup;
- maximum source and target length 256;
- training seeds 13, 42, and 73; and
- the same final-checkpoint selection and beam-search settings for every system.

Every arm would have the same number of examples, epochs, optimizer steps, and padded tensor shapes. I would also report source words, target subword tokens, non-padding tokens, runtime, and peak GPU memory. This matters because “equal examples” does not guarantee equal useful-token exposure.

I would not use domain tags or in-task target-language learning in the primary comparison because they would introduce an additional intervention. They are better treated as later ablations.

## 3. Questions and hypotheses

I would write down the following hypotheses before training:

1. **Language transfer:** multilingual adaptation should help Gujarati, Georgian, and Tamil news most clearly because those targets receive direct supervision.
2. **Domain robustness:** multi-domain adaptation should help Mandarin social, speech, and literary translation most clearly.
3. **Matched-condition specialization:** the Mandarin-news control may perform best on Mandarin news because all of its adaptation data reinforce that condition.
4. **Quality and faithfulness may disagree:** the system with the best chrF++ or BLEU score may still omit facts, change entities or numbers, or respond poorly to controlled source changes.
5. **There may be no single winner:** multilingual and multi-domain adaptation solve different problems, so the comparison should be reported by language and domain rather than collapsed into one score.

These are expected patterns, not claimed findings.

## 4. Evaluation plan

### 4.1 Translation quality

I would make **chrF++** the primary automatic metric because it is character based and can be applied consistently across the four writing systems [8]. I would also report **SacreBLEU** separately for each language and domain, including its tokenizer and complete signature [9]. I would not average BLEU across scripts.

COMET could be included as a secondary metric if time permits, but I would not make it central because learned metrics can have uneven validity across low-resource languages and specialized domains [10,11].

### 4.2 Source faithfulness

Standard overlap scores do not directly show whether a translation follows its source. I would therefore add three small diagnostics:

- **Number and date preservation:** recall of source-side numbers, dates, and units, plus the rate of numbers introduced without source support.
- **Mandarin bilingual audit:** two annotators would review the same outputs for unsupported addition, omission, contradiction, entity or number mutation, non-translation, repetition, and fluency-only errors. Each error would receive minor, major, or critical severity.
- **Counterfactual sensitivity:** selected Mandarin sources would be changed in exactly one fact, such as a date, number, entity, or polarity. A successful model should express the new fact, remove the old fact, and otherwise preserve the sentence’s meaning.

The number diagnostic is intentionally narrow and would not replace human review. For example, correctly copying “12%” does not prove that the model translated the surrounding claim correctly.

### 4.3 Comparison and uncertainty

Each adapted system would be compared with the Mandarin-news control on the relevant cells. I would report all three seeds and use paired document-level bootstrap resampling with 1,000 replicates for confidence intervals. I would avoid significance claims if the samples or document counts are too small.

I would not report a single aggregate winner because language, corpus, reference style, script, and direct supervision differ across cells. A defensible conclusion would look like:

> Under this checkpoint, data, and budget, multilingual allocation improved particular language-transfer cells, while multi-domain allocation improved particular Mandarin domain-transfer cells, with the following changes in source-faithfulness errors.

## 5. How I would interpret possible results

If the multilingual system improves Gujarati, Georgian, and Tamil news but not Mandarin domain shifts, that would support the value of direct cross-language supervision under this budget. If the multi-domain system improves Mandarin social or speech translation, that would support domain diversity for a fixed language. Strong Mandarin-news performance from the control would show the cost of spreading a small budget across heterogeneous conditions.

Negative transfer would also be informative. NLLB shares limited capacity across many languages, so adding diverse examples could reduce performance on a matched condition. I would examine whether this is associated with script, sentence length, domain, or target-token exposure rather than treating it as random noise.

Most importantly for HALO, I would compare metric gains with the faithfulness diagnostics. A system that produces more reference-like translations but more entity mutations or weaker counterfactual sensitivity would not be an unqualified improvement.

## 6. Small supporting code demonstration

To confirm that the plan is executable, I created one Colab notebook that loads the pinned NLLB checkpoint, validates the datasets and language tokens, translates deterministic samples, computes chrF++ and SacreBLEU, and exports report-ready files. It runs **zero-shot inference only**; the LoRA comparison above remains proposed work.

The notebook produced 190 nonempty translations:

- 25 NTREX examples for each of the four target languages;
- 15 Mandarin examples from each WMT24++ auxiliary domain;
- 25 TICO Mandarin medical examples; and
- 10 Mandarin counterfactual pairs, producing 20 translations.

| Zero-shot test cell | n | chrF++ | BLEU | Source-number recall |
|---|---:|---:|---:|---:|
| NTREX Gujarati news | 25 | 48.01 | 18.10 | 14/17 |
| NTREX Georgian news | 25 | 47.07 | 18.24 | 13/17 |
| NTREX Tamil news | 25 | 45.97 | 11.64 | 14/17 |
| NTREX Mandarin news | 25 | 22.77 | 29.69 | 12/17 |
| TICO Mandarin medical | 25 | 34.62 | 40.51 | 37/41 |
| WMT Mandarin social | 15 | 28.16 | 22.82 | 2/6 |
| WMT Mandarin speech | 15 | 17.46 | 23.45 | 12/15 |
| WMT Mandarin literary | 15 | 12.13 | 13.15 | 1/8 |

These results compare different zero-shot conditions, **not** the adaptation strategies. Their value is to show that the end-to-end workflow works and that the domains have visibly different difficulty and error patterns. Small denominators and different reference conventions prevent broad rankings.

I also reviewed 20 Mandarin outputs: 10 news and 10 medical. The audit found no unsupported additions, but it identified 8 omissions, 6 contradictions, and 5 entity or number mutations; categories can overlap. Six of ten counterfactual pairs passed all criteria. The failures included an omitted percentage, an incorrectly rendered monetary amount, and entity changes that destabilized unrelated wording. This small sample is illustrative rather than an estimate of the model’s general hallucination rate.

The [executable Colab notebook](https://colab.research.google.com/github/EricWcr7/multilingual-vs-multidomain-nmt/blob/codex/halo-modern-pilot/notebooks/halo_option1_demo.ipynb) and [reproducibility repository](https://github.com/EricWcr7/multilingual-vs-multidomain-nmt/tree/codex/halo-modern-pilot) contain the exact sample IDs, predictions, metric signatures, audit labels, dependency versions, model revision, and file hashes. A full run was completed on both CPU and a Colab T4; because a few generated strings differed across devices, the committed metrics consistently use the recorded CPU run rather than mixing results.

## 7. Limitations and connection to HALO

This design cannot isolate language diversity from every other factor. Language, corpus, script, reference style, training exposure, and document count are partly confounded. Equal example counts do not equal equal information or compute. WMT24++ literary data are especially small, TICO is only one medical test, and the demonstration has one Mandarin annotator. Pretraining overlap is unknown.

The connection to HALO is deliberately limited:

> Translation is a controlled conditional-generation setting in which the source is the grounding context. The study tests sensitivity and faithfulness to that context across languages and domains; it does not establish that translation hallucinations generalize to open-world LLM hallucinations or evaluate a HALO mitigation method.

If the proposed experiment were completed, the next step would be to select a clearly observed failure pattern and test one HALO-relevant intervention, such as multilingual grounding regularization, cross-lingual calibration, or sensitivity-based filtering. That would be a separate experiment with its own baseline and evaluation.

## 8. Proposed deliverables

For this challenge, I would submit:

1. this experimental plan;
2. the small executable notebook as feasibility evidence;
3. saved predictions, metrics, audit labels, and examples; and
4. clear instructions for reproducing the demonstration and extending it to the proposed LoRA study.

The plan is the main answer. The code is included only to show that the data and evaluation pipeline are concrete enough to run.

## References

1. Meta AI. “NLLB-200 Distilled 600M” model card. <https://huggingface.co/facebook/nllb-200-distilled-600M>
2. NLLB Team et al. “No Language Left Behind: Scaling Human-Centered Machine Translation.” *arXiv:2207.04672*, 2022. <https://arxiv.org/abs/2207.04672>
3. Deutsch, D. et al. “WMT24++: Expanding the Language Coverage of WMT24 to 55 Languages & Dialects.” *arXiv:2502.12404*, 2025. <https://arxiv.org/abs/2502.12404>
4. Federmann, C., Kocmi, T., and Xin, Y. “NTREX-128 — News Test References for MT Evaluation of 128 Languages.” *SUMEval 2022*. <https://aclanthology.org/2022.sumeval-1.4/>
5. Anastasopoulos, A. et al. “TICO-19: The Translation Initiative for COvid-19.” *NLP-COVID19 at EMNLP 2020*. <https://aclanthology.org/2020.nlpcovid19-2.5/>
6. TICO-19 project. Official dataset. <https://tico-19.github.io/>
7. Goyal, N. et al. “The FLORES-101 Evaluation Benchmark for Low-Resource and Multilingual Machine Translation.” *TACL* 10, 2022; FLORES+ portal. <https://huggingface.co/datasets/openlanguagedata/flores_plus>
8. Popović, M. “chrF: Character n-gram F-score for Automatic MT Evaluation.” *WMT 2015*. <https://aclanthology.org/W15-3049/>
9. Post, M. “A Call for Clarity in Reporting BLEU Scores.” *WMT 2018*. <https://aclanthology.org/W18-6319/>
10. Rei, R. et al. “COMET: A Neural Framework for MT Evaluation.” *EMNLP 2020*. <https://aclanthology.org/2020.emnlp-main.213/>
11. Unbabel. “wmt22-comet-da” model card. <https://huggingface.co/Unbabel/wmt22-comet-da>
12. Hu, E. J. et al. “LoRA: Low-Rank Adaptation of Large Language Models.” *ICLR 2022*. <https://openreview.net/forum?id=nZeVKeeFYf9>
