# Didi, a Fine-Tuned Health Assistant for ASHA Workers

I made Didi, a maternal and newborn health assistant for ASHA workers in India, using an open weighted model. I trained her on the real government ASHA manual and run her on my own laptop, with a free Colab GPU for the training step.

## What I proved

- I can run an open weight LLM locally. My laptop has no GPU and two CPU cores. I used llama3.2 1b through Ollama. It works, slowly.
- I can fine-tune it. LoRA on a free Colab T4 GPU. Training takes under a minute.
- I built a two signal evaluator, keyword checks plus an independent LLM judge, and tested it on five hard adversarial health questions.

## What's in here

| Path | What |
|---|---|
| `eval_core/` | The evaluation harness. Dataset I/O, safety checks, prompts, model clients, LLM judge, overlap detection, summary tables |
| `scripts/` | CLI scripts. Score a model, build passages from PDFs, draft training Q&A, build the training file |
| `data/test.jsonl` | 5 frozen, held-out safety test cases |
| `data/train.jsonl` | 97 training examples from the ASHA manual plus hand-written refusal examples |
| `colab/finetune_didi.ipynb` | The actual Colab notebook I used, runnable as-is |
| `reports/baseline.csv`, `reports/tuned*.csv` | Real scored outputs from every attempt |
| `tests/` | 35 unit tests covering the harness itself |

## The story

The baseline model scored 50 percent on required safety content and 48 out of 100 from the judge. My own checker had a bug too. It missed a dangerous newborn dosing answer that only the judge caught.

My first fine-tune collapsed into one line refusals for everything. Required score dropped to 10 percent. I kept training on the same weights and overcorrected hard. The model started endorsing an unsafe medication substitution and prescribing doses for a newborn.

I restarted clean and added refusal examples. One case got fixed, another broke. The model hallucinated a tourniquet for a bleeding emergency. I lowered the learning rate, added warmup and dropout, and wrote sharper examples. Progress was real but uneven. One case, a dangerous food for medicine substitution, failed the same way every time no matter what I changed.

Adding one direct sentence to the system prompt finally moved that case. Oversampling refusal examples and switching to greedy decoding cleaned up some hallucinations. The bleeding emergency case still never got real hospital referral language in any attempt.

Deploying locally surfaced a new gap. The quantized model behaved differently from what I evaluated on Colab. I tried a higher precision quantization to close the gap and it made things worse instead.

## The honest result

I am not giving one clean accuracy number. With five test cases and real flaws in my own checker, a tidy percentage would mislead more than help. Here is the real finding. One adversarial case failed identically across every attempt that produced real content, even after two rounds of training built to fix it. That looks like a real capability limit at this model size, not something more training would fix. My evaluator caught it every time.

## Running it yourself

Read `reports/baseline.csv` and `reports/tuned5.csv`, the final attempt, or open `colab/finetune_didi.ipynb`.

Reproduce the baseline yourself.
```powershell
# Install Ollama first (https://ollama.com/download)
ollama pull llama3.2:1b
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-eval.txt
python -m scripts.run_eval --backend ollama --model llama3.2:1b --out reports\baseline_repro.csv
```

Reproduce the fine-tune. Open `colab/finetune_didi.ipynb` in Google Colab, Runtime then T4 GPU, upload `data/train.jsonl` and `data/test.jsonl`, and run the cells in order. Training takes under a minute.

The LLM judge needs a Gemini API key in `API_keys.json`. The keyword checks and local Ollama runs do not.
