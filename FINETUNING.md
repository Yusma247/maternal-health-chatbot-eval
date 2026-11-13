# Didi: fine-tuning an open-weight model for ASHA workers

I wanted to answer a simple question honestly: can I actually fine-tune an
open-weight model myself, not just call an API, and prove it got better at
something real? This is that project. I built Didi, a maternal and newborn
health assistant for ASHA workers in India, trained on the government
training manual they actually use, and I run the whole thing on my own
laptop plus a free Colab GPU for the training step itself.

## What I actually proved

- I can run an open-weight LLM locally. My laptop has no GPU and two CPU
  cores, so I picked `llama3.2:1b` and ran it through Ollama. It works,
  slowly, but it works.
- I can fine-tune it. LoRA on a free Colab T4, training takes under a
  minute once the data's ready.
- The harder part was proving it actually got *better*, not just
  different. For that I built a two-signal evaluator - keyword checks plus
  an independent LLM judge - and tested it against five adversarial
  maternal and newborn health questions I deliberately picked to be hard
  to get right.

## What's in here

| Path | What |
|---|---|
| `eval_core/` | The evaluation harness: dataset I/O, safety checks, prompts, model clients, LLM judge, overlap/leak detection, summary tables |
| `scripts/` | CLI scripts: score a model (`run_eval.py`), build passages from PDFs, draft training Q&A, build the training file |
| `data/test.jsonl` | 5 frozen, held-out safety test cases (adversarial maternal/newborn health prompts) |
| `data/train.jsonl` | 97 training examples built from government ASHA manual passages, plus hand-written refusal/safety examples |
| `colab/finetune_didi.ipynb` | The actual Colab notebook I used to fine-tune - runnable as-is |
| `reports/baseline.csv`, `reports/tuned*.csv` | Real scored outputs from every attempt |
| `tests/` | 35 unit tests covering the harness itself |

## The actual story

I started with a baseline score: the untouched model got 50% on my
required-content checks and a judge score of 48/100. Right away I found a
bug in my own checker - it missed a genuinely dangerous answer (bad dosing
advice for a newborn) that only the LLM judge caught. That was the first
sign a single metric wasn't going to tell me the whole story here.

My first real fine-tuning attempt went badly in a way I didn't expect.
Instead of getting better or worse at giving advice, the model stopped
giving advice at all. It collapsed into one-line refusals for everything -
"I can't give you medical advice," full stop. Required score dropped to
10%. Not what I wanted, but at least it was a clear signal something in the
training setup was off.

I kept training on the same weights to try to recover it, and
overcorrected hard - the model swung to confidently endorsing an unsafe
medication substitution and prescribing specific doses for a newborn. That
was worse than the collapse. Fine-tuning a tiny model turned out to be more
of a knife-edge than I expected.

From there it became genuine debugging. I restarted clean, added refusal
examples to the training data, and watched one case get fixed while a
different one broke - this time the model hallucinated advice to use a
tourniquet for a bleeding emergency. I lowered the learning rate, added a
warmup schedule and some dropout to fight overfitting, and wrote more
targeted training examples. Progress was real but inconsistent - some cases
moved, and one case (a dangerous food-for-medicine substitution) failed the
exact same way every single time, no matter what I changed in the data.

The one thing that actually moved that stuck case was blunt: I added an
explicit sentence to the system prompt naming the exact failure pattern.
Not elegant, but it worked, and it taught me something - sometimes the fix
isn't more training data, it's just telling the model directly what it
keeps missing.

Even after oversampling the refusal examples and switching to greedy
decoding to cut down on hallucinated specifics, one case never fully
resolved - a bleeding emergency that should get real hospital-referral
language, and never did, across every attempt I tried.

Then I deployed it locally and found a gap I wasn't expecting at all - the
quantized model I could actually run on Ollama behaved differently from the
one I'd evaluated on Colab. I tried a higher-precision quantization hoping
to close that gap, and it made things measurably worse instead. That one
surprised me the most, honestly - I'd assumed more precision would just
mean closer-to-original behavior, not a new failure mode of its own.

## The honest result

I'm not going to give you one clean accuracy number, because with five test
cases and some real false-positive and false-negative issues in my own
checker (which I found and partially fixed along the way), a tidy
percentage would be more misleading than useful. The finding that actually
matters: one adversarial case failed identically across every attempt that
produced real content, including two rounds of training data aimed
specifically at fixing it. At this model size and dataset size, that looks
like a real capability ceiling, not something I could train my way past.
Catching that, reliably and repeatedly, is what the evaluator was for.

## Running it yourself

**See the results without installing anything:** read `reports/baseline.csv`
and `reports/tuned5.csv` (the final, cleanest attempt), or open
`colab/finetune_didi.ipynb` directly on GitHub.

**Reproduce the baseline yourself:**
```powershell
# Install Ollama (https://ollama.com/download), then:
ollama pull llama3.2:1b
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-eval.txt
python -m scripts.run_eval --backend ollama --model llama3.2:1b --out reports\baseline_repro.csv
```

**Reproduce the fine-tune:** open `colab/finetune_didi.ipynb` in Google
Colab (Runtime > T4 GPU), upload `data/train.jsonl` and `data/test.jsonl`,
and run the cells in order. Training takes under a minute.

The LLM judge (`eval_core/judge.py`) needs a Gemini API key
(`API_keys.json`, see `eval_core/models.py::load_gemini_keys` for the
format) - the keyword checks and local Ollama runs don't.
