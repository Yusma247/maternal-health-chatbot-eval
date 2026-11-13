# Run each "# %%" block as one Colab cell. Runtime > Change runtime type > T4 GPU.
# Upload train.jsonl and test.jsonl to the Colab Files panel first.

# %% 1. Install (if this fails, copy the install cell from Unsloth's current Colab notebook)
# !pip install unsloth

# %% 2. Settings. The base model must match the Ollama baseline (llama3.2:1b).
# Check the model licence first before any use beyond this demo.
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
SYSTEM = (
    "You are Didi, a maternal and child health assistant for ASHA workers "
    "in rural India. You give accurate, safe health information about pregnancy, "
    "antenatal care, nutrition, and newborn care. "
    "You always respond in English. "
    "You never give dangerous medical advice. "
    "You always refer critical cases to a doctor or hospital immediately."
)
MAX_SEQ = 2048

# %% 3. Load the model in 4 bit and add LoRA adapters
from unsloth import FastLanguageModel

model, tok = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# %% 4. Load train.jsonl and turn each chat into one training text
from datasets import load_dataset

ds = load_dataset("json", data_files="train.jsonl", split="train")
ds = ds.map(lambda ex: {"text": tok.apply_chat_template(
    ex["messages"], tokenize=False, add_generation_prompt=False)})
print(len(ds), "examples")
print(ds[0]["text"][:600])

# %% 5. Train. If an argument name errors, copy the trainer cell from Unsloth's
# current notebook and keep the dataset cell above.
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        output_dir="outputs",
        report_to="none",
        seed=3407,
    ),
)
trainer.train()

# %% 6. Save the adapter to Drive so a session reset does not lose it
from google.colab import drive

drive.mount("/content/drive")
model.save_pretrained("/content/drive/MyDrive/didi-lora")
tok.save_pretrained("/content/drive/MyDrive/didi-lora")

# %% 7. Answer the test prompts here and save answers_tuned.csv (the safe fallback)
import csv
import json

FastLanguageModel.for_inference(model)
rows = []
for line in open("test.jsonl", encoding="utf-8"):
    case = json.loads(line)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": case["prompt"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**inputs, max_new_tokens=400, temperature=0.2, do_sample=True)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    rows.append({"id": case["id"], "answer": answer.strip()})

with open("answers_tuned.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "answer"])
    writer.writeheader()
    writer.writerows(rows)
print("Saved answers_tuned.csv. Download it from the Files panel.")

# %% 8. Export for Ollama. Look for the .gguf file in the folder it prints, then download it.
model.save_pretrained_gguf("didi-ft", tok, quantization_method="q4_k_m")
