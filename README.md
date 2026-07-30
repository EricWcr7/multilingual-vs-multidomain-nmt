# Multilingual vs. Multi-domain NMT

This repository is the reproducibility package for the HALO undergraduate
challenge response:

> **Multilingual vs. Multi-domain NMT: Experimental Plan**

It contains two deliberately separate contributions:

1. a **proposed, not executed** equal-budget LoRA comparison of multilingual
   and multi-domain adaptation; and
2. a small **executed zero-shot demonstration** for English→Gujarati,
   Georgian, Tamil, and Simplified Chinese that checks the feasibility of the
   data and evaluation pipeline.

The demonstration uses
[`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
at revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`. NLLB is used
as a compact experimental instrument, not presented as the state of the art
in 2026. Its checkpoint is licensed CC BY-NC 4.0 and is used here only for
non-commercial research.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EricWcr7/multilingual-vs-multidomain-nmt/blob/codex/halo-modern-pilot/notebooks/halo_option1_demo.ipynb)

[Read the undergraduate challenge question response (PDF)](undergraduate_challenge_question_response.pdf)

## Plan at a glance

The proposed study asks whether a fixed adaptation budget is better spent
across target languages or across domains. Every adapted system starts from
the same NLLB checkpoint, uses the same LoRA configuration, and receives 308
training examples. The comparison includes:

- an unadapted baseline;
- a Mandarin-news control that receives more data without added diversity;
- a multilingual system trained on news in Mandarin, Gujarati, Georgian, and
  Tamil; and
- a multi-domain system trained on Mandarin news, social, speech, and literary
  text.

The plan uses document-level splits and source hashing to prevent leakage.
It evaluates translation quality with chrF++ and SacreBLEU, then checks source
faithfulness through number and date preservation, a bilingual Mandarin audit,
and controlled source changes. Three training seeds and document-level
bootstrap confidence intervals are proposed so that results can be reported
by language and domain rather than reduced to a single winner.

The LoRA comparison is an experimental proposal and has not been run. The
notebook in this repository is a smaller, executed zero-shot demonstration
that verifies the data, inference, and evaluation pipeline needed to support
that plan.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── undergraduate_challenge_question_response.pdf
├── notebooks/
│   └── halo_option1_demo.ipynb
├── report/
│   └── halo_option1_answer.md
└── results/
    ├── demo_manifest.json
    ├── demo_predictions.csv
    ├── demo_metrics.csv
    ├── mandarin_audit.csv
    └── examples.md
```

Raw datasets and model checkpoints are downloaded to temporary storage and
are not committed.

## How the code works

The implementation is contained in
[`notebooks/halo_option1_demo.ipynb`](notebooks/halo_option1_demo.ipynb).
Its cells form one reproducible pipeline:

1. **Pin the environment.** The notebook installs exact package versions,
   records the Python, PyTorch, and hardware configuration, and fixes all
   random seeds.
2. **Verify inputs.** It downloads pinned revisions of NTREX-128, WMT24++,
   TICO-19, and NLLB, then checks the expected SHA-256 hash of every data and
   model file before using it.
3. **Prepare the data.** Text is normalized and deduplicated, invalid WMT24++
   rows are removed, and deterministic document-level splits are created.
   Assertions check schemas, row counts, split isolation, source-hash leakage,
   token limits, and target-language mappings.
4. **Define the experiment and sample the demonstration.** The code records
   the equal-budget LoRA arm manifests described in the plan, but does not
   train them. It separately chooses a deterministic zero-shot sample for the
   executable demonstration.
5. **Run inference.** The pinned NLLB checkpoint translates groups of English
   sources into Gujarati, Georgian, Tamil, or Simplified Chinese with
   four-beam decoding. The code verifies each language token and forced
   beginning-of-sentence token before generation.
6. **Evaluate and export.** It computes chrF++ and language-appropriate
   SacreBLEU scores, runs numeric source-faithfulness and Mandarin
   counterfactual checks, prepares the manual audit worksheet, and writes the
   five public artifacts in `results/`.

The optional `HALO_REDUCED_DEMO=1` environment variable selects a smaller
smoke-test sample. `HALO_LOCAL_MODEL_DIR` can reuse a local checkpoint only
when all expected model files match the pinned hashes. COMET is included as
an optional, disabled secondary metric; it is not required for the reported
results.

## Demo results

The executed demonstration produced 190 nonempty zero-shot translations.
The aggregate results for its eight evaluation cells are:

| Test condition | n | chrF++ | BLEU | Source-number recall |
|---|---:|---:|---:|---:|
| Gujarati news | 25 | 48.01 | 18.10 | 82.4% |
| Georgian news | 25 | 47.07 | 18.24 | 76.5% |
| Tamil news | 25 | 45.97 | 11.64 | 82.4% |
| Mandarin news | 25 | 22.77 | 29.69 | 70.6% |
| Mandarin medical | 25 | 34.62 | 40.51 | 90.2% |
| Mandarin social | 15 | 28.16 | 22.82 | 33.3% |
| Mandarin speech | 15 | 17.46 | 23.45 | 80.0% |
| Mandarin literary | 15 | 12.13 | 13.15 | 12.5% |

The scores show that zero-shot performance varies substantially by language
and domain. Mandarin medical produced the strongest reported Mandarin scores,
while Mandarin literary was the weakest condition. These cells use different
corpora, sample sizes, and reference styles, so they should not be treated as
a controlled ranking.

A manual review of 20 Mandarin outputs found 8 omissions, 6 contradictions,
and 5 entity or number mutations; these categories can overlap. Six of the 10
Mandarin counterfactual pairs passed all sensitivity criteria. The failures
included dropped or incorrectly rendered facts and unrelated wording changes.

The result establishes that the pinned data, inference, evaluation, and export
pipeline runs end to end. It does **not** determine whether multilingual or
multi-domain adaptation is better, because the proposed LoRA comparison has
not been trained.

## Run in Colab

1. Open the badge above.
2. Select **Runtime → Change runtime type → T4 GPU** (or another CUDA GPU).
3. Run all cells in order.
4. Download the generated `halo_option1_results.zip` file when prompted.

The full 190-translation configuration was also verified top-to-bottom on a
Colab Tesla T4 on July 29, 2026. All 13 code cells completed in order and the
five expected result files plus the ZIP were produced. The committed result
files are the separately recorded CPU run described in the report. The CPU
and CUDA runs used the same verified seven-file model fingerprint and inputs,
but their predictions and derived metrics were not byte-identical; results are
therefore reported from one device-specific run rather than mixed.

The notebook validates revisions, checksums, schemas, document-level splits,
source-hash leakage, language tokens, deterministic sample IDs, output
non-emptiness, and metric fixtures before writing artifacts.

## Run locally

Python 3.11–3.13 is recommended. A GPU is helpful but not required for a
reduced smoke run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter execute notebooks/halo_option1_demo.ipynb
```

The notebook writes public artifacts to `results/` when run from the repository
root. Public files contain IDs, predictions, scores, and metadata—not bulk
source text or references.

## Data and evaluation

- [NTREX-128](https://github.com/MicrosoftTranslator/NTREX), revision
  `468c6b69c7f6a75d31d4743d9daba2af566cc18d` (CC BY-SA 4.0)
- [WMT24++](https://huggingface.co/datasets/google/wmt24pp), revision
  `e65f5856b1de3319c748c15e5aec0bc2336ec3b0` (Apache 2.0)
- [TICO-19](https://tico-19.github.io/) official test archive (translations
  released under CC0; source licenses vary by upstream row, and the public
  audit output omits source/reference text)

chrF++ is the primary automatic metric. SacreBLEU is reported separately per
language/domain with its tokenizer and full signature. Digit-based
source-faithfulness diagnostics and a small Mandarin counterfactual check are
descriptive; they do not establish a general hallucination rate.

## Interpretation boundary

Translation is a controlled conditional-generation setting in which the
source is the grounding context. This work tests sensitivity and faithfulness
to that context across languages and domains. It does not establish that
translation hallucinations generalize to open-world LLM hallucinations, and it
does not evaluate a HALO mitigation method.
