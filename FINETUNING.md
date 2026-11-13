# Didi Fine-Tuning Demo

Fine-tunes an open-weight model (`llama3.2:1b`) into **Didi**, a maternal and
newborn health assistant for ASHA workers, trained on real government ASHA
training material, and evaluated with an independent safety harness before
and after tuning.

## What this proves

- An open-weight LLM runs locally (Ollama, on a 2-core/8GB laptop with no GPU).
- A LoRA fine-tune runs on a free Colab T4 GPU in under a minute.
- A two-signal safety evaluator (keyword checks + an independent LLM judge)
  catches real, specific safety failures across the fine-tuning process -
  not just an aggregate score.

## What's in here

| Path | What |
|---|---|
| `eval_core/` | The evaluation harness: dataset I/O, safety checks, prompts, model clients, LLM judge, overlap/leak detection, summary tables |
| `scripts/` | CLI scripts: score a model (`run_eval.py`), build passages from PDFs, draft training Q&A, build the training file |
| `data/test.jsonl` | 5 frozen, held-out safety test cases (adversarial maternal/newborn health prompts) |
| `data/train.jsonl` | 97 training examples built from government ASHA manual passages, plus hand-written refusal/safety examples |
| `colab/finetune_didi.ipynb` | The actual Colab notebook used to fine-tune - runnable as-is |
| `reports/baseline.csv`, `reports/tuned*.csv` | Real scored outputs from every attempt |
| `tests/` | 35 unit tests covering the harness itself |

## The journey (what I changed, and why)

1. **Baseline**: `llama3.2:1b` scored 50% on required safety content, judge
   score 48/100. Found a real gap in the keyword checker itself along the
   way - it missed a dangerous newborn-dosing answer that only the LLM judge
   caught.
2. **First fine-tune attempt** (67 examples, 2 epochs): collapsed into bare
   one-line refusals, losing all useful content. Required score dropped to 10%.
3. **More training** (continued on the same weights): over-corrected into
   dangerous compliance - endorsed an unsafe medication substitution,
   prescribed a specific dose for a newborn.
4. **Fresh training + refusal examples**: fixed one case, broke another
   (hallucinated a tourniquet for a bleeding emergency). Diagnosed likely
   cause: an undertrained model generalizing a learned response *style*
   without the underlying safety judgment behind it.
5. **Hyperparameter fixes** (lower LR, warmup, LoRA dropout) + sharper,
   more targeted refusal examples: partial, inconsistent progress - some
   cases fixed, one case failed identically across every single attempt
   that produced real content.
6. **Explicit system-prompt instruction** naming the exact failure pattern:
   first real, genuine movement on the case that had never budged.
7. **Oversampled the refusal examples + switched to greedy decoding**:
   cleaned up some hallucinations, but one case (a bleeding emergency never
   getting real hospital-referral content) never resolved across any
   attempt.
8. **Deployed locally via Ollama (GGUF export)**: found the exported,
   quantized model behaves differently from what was evaluated on Colab -
   a real, disclosed gap between "what I tested" and "what runs locally."
   Tried a higher-precision quantization (Q8_0) to close the gap; it made
   things measurably worse instead, which is itself a finding, not a fix.

## The honest result

There is no single clean "accuracy" number here, deliberately - with only 5
test cases, and known measurement artifacts in the keyword checker (caught
and partially fixed along the way), a percentage would overstate precision
that doesn't exist. The real, defensible finding: **one adversarial case
(a dangerous food-for-medicine substitution) failed identically across every
fine-tuning attempt that produced real content, including after two rounds
of training data built specifically to fix it.** That's a genuine capability
limit of a 1B-parameter model trained on ~100 examples, not a bug to patch
away - and it's exactly the kind of thing this evaluation harness was built
to catch.

## Running it yourself

**See the results without installing anything:** read `reports/baseline.csv`
and `reports/tuned5.csv` (the final, cleanest attempt), or open
`colab/finetune_didi.ipynb` directly on GitHub.

**Reproduce the baseline:**
```powershell
# Install Ollama (https://ollama.com/download), then:
ollama pull llama3.2:1b
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-eval.txt
python -m scripts.run_eval --backend ollama --model llama3.2:1b --out reports\baseline_repro.csv
```

**Reproduce the fine-tune:** open `colab/finetune_didi.ipynb` in Google
Colab (Runtime > T4 GPU), upload `data/train.jsonl` and `data/test.jsonl`,
run the cells in order. Training takes under a minute.

The LLM judge (`eval_core/judge.py`) needs a Gemini API key
(`API_keys.json`, see `eval_core/models.py::load_gemini_keys` for the
format) - the keyword checks and local Ollama runs don't.
