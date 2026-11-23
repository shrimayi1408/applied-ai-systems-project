"""Fine-tune flan-t5-small on the DJ Nova voice data. Usage: python -m src.train_dj"""
import json
import random
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

BASE = "google/flan-t5-small"
OUT = "models/dj_nova"
EPOCHS, BATCH, LR, SEED = 4, 16, 3e-4, 0


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
    data = read("data/dj/train.jsonl")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()
    for epoch in range(EPOCHS):
        random.shuffle(data)
        total = 0.0
        for i in range(0, len(data), BATCH):
            b = data[i:i + BATCH]
            enc = tok([x["input"] for x in b], return_tensors="pt", padding=True, truncation=True, max_length=128)
            lab = tok([x["target"] for x in b], return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids
            lab[lab == tok.pad_token_id] = -100
            loss = model(**enc, labels=lab).loss
            loss.backward(); opt.step(); opt.zero_grad()
            total += loss.item() * len(b)
        print(f"epoch {epoch + 1}/{EPOCHS} loss={total / len(data):.4f}", flush=True)
    model.save_pretrained(OUT); tok.save_pretrained(OUT)
    print(f"saved fine-tuned model -> {OUT}")


if __name__ == "__main__":
    main()
