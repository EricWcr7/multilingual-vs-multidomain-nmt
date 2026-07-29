# Multilingual vs. Multi-domain NMT

This repository is the reproducibility package for the HALO undergraduate
challenge response:

> **Multilingual vs. Multi-domain NMT: A Controlled Experimental Plan with
> Multilingual Grounding Diagnostics**

It contains two deliberately separate contributions:

1. an **executed zero-shot demonstration** for English→Gujarati, Georgian,
   Tamil, and Simplified Chinese; and
2. a **proposed, not executed** equal-budget LoRA comparison of multilingual
   and multi-domain adaptation.

The demonstration uses
[`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)
at revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`. NLLB is used
as a compact experimental instrument, not presented as the state of the art
in 2026. Its checkpoint is licensed CC BY-NC 4.0 and is used here only for
non-commercial research.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EricWcr7/multilingual-vs-multidomain-nmt/blob/codex/halo-modern-pilot/notebooks/halo_option1_demo.ipynb)

[Read the six-page submission report (PDF)](report/halo_option1_answer.pdf)

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── halo_option1_demo.ipynb
├── report/
│   ├── halo_option1_answer.md
│   └── halo_option1_answer.pdf
└── results/
    ├── demo_manifest.json
    ├── demo_predictions.csv
    ├── demo_metrics.csv
    ├── mandarin_audit.csv
    └── examples.md
```

Raw datasets and model checkpoints are downloaded to temporary storage and
are not committed.

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
