import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
)

from sacrebleu import corpus_bleu
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(f"PyTorch: {torch.__version__}")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Compute capability may vary; mixed precision can speed up training.")



# ---- Experiment config ----
MODEL_NAME = "t5-small"           # ssmall version of T5 for faster training
SRC_LANG = "English"
TGT_LANG = "French"
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 2                     # keep small for speed
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06                # % of total steps used for LR warmup
GRAD_ACCUM_STEPS = 2               # simulate larger batch on limited GPU
MAX_NEW_TOKENS = 64                # generation length for eval/inference
NUM_BEAMS = 4                      # beam search width for generation

# Subsampling for fast classroom runs (set to None to use full sets)
SUBSET_TRAIN = 4000
SUBSET_VALID = 500

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using DEVICE:", DEVICE)

# Where to save figures/metrics
OUT_DIR = Path("t5_runs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
    

# We load the english-french portion of the OPUS Books dataset
raw_dset = load_dataset("opus_books", "en-fr")
print(raw_dset)

VALIDATION_RATIO = 0.2
split = raw_dset["train"].train_test_split(test_size=VALIDATION_RATIO, seed=SEED)
train_raw = split["train"]
valid_raw = split["test"]

# Apply subsampling if configured
if SUBSET_TRAIN is not None:
    train_raw = train_raw.shuffle(seed=SEED).select(range(min(SUBSET_TRAIN, len(train_raw))))
if SUBSET_VALID is not None:
    valid_raw = valid_raw.shuffle(seed=SEED).select(range(min(SUBSET_VALID, len(valid_raw))))

print("Train size:", len(train_raw), "| Valid size:", len(valid_raw))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

prefix = f"translate {SRC_LANG} to {TGT_LANG}: "

def preprocess_examples(batch):
    src_texts = [prefix + ex["en"] for ex in batch["translation"]]
    tgt_texts = [ex["fr"] for ex in batch["translation"]]
    model_inputs = tokenizer(src_texts, max_length=MAX_SOURCE_LENGTH, truncation=True)
    labels = tokenizer(tgt_texts, max_length=MAX_TARGET_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map with batched=True returns lists of lists (no padding yet)
train_tokenized = train_raw.map(preprocess_examples, batched=True, remove_columns=train_raw.column_names)
valid_tokenized = valid_raw.map(preprocess_examples, batched=True, remove_columns=valid_raw.column_names)

print(train_tokenized)
print(valid_tokenized)

def collate_fn(features):
    input_batch = {"input_ids": [f["input_ids"] for f in features],
                   "attention_mask": [f["attention_mask"] for f in features]}
    labels_batch = {"input_ids": [f["labels"] for f in features]}

    padded_inputs = tokenizer.pad(input_batch, padding=True, return_tensors="pt")
    padded_labels = tokenizer.pad(labels_batch, padding=True, return_tensors="pt")["input_ids"]
    # Replace padding token id with -100 to ignore in loss computation
    padded_labels = padded_labels.masked_fill(padded_labels == tokenizer.pad_token_id, -100)
    padded_inputs["labels"] = padded_labels
    return padded_inputs

train_loader = DataLoader(train_tokenized, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_tokenized, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

len(train_loader), len(valid_loader)

# We load the pretrained T5 model for conditional generation
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_update_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
t_total_steps = NUM_EPOCHS * num_update_steps_per_epoch
num_warmup_steps = int(WARMUP_RATIO * t_total_steps)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=t_total_steps,
)

print(f"Total steps: {t_total_steps} | Warmup steps: {num_warmup_steps}")


# WandB – Initialize a new run
wandb.init(project="t5_translation", settings=wandb.Settings(_service_wait=300, init_timeout=300))


# Log metrics with wandb
wandb.watch(model, log="all")


best_valid_bleu = -1.0
train_loss_history, valid_loss_history, bleu_history = [], [], []

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader, start=1), total=len(train_loader), desc=f"Epoch {epoch} [train]")

    optimizer.zero_grad(set_to_none=True)
    for step, batch in pbar:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        outputs = model(**batch)
        # Scales the loss before backpropagation so that gradient accumulation matches the magnitude of a larger effective batch
        loss = outputs.loss / GRAD_ACCUM_STEPS
        loss.backward()

        # Only update weights and step scheduler every GRAD_ACCUM_STEPS to simulate larger batch size
        if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        wandb.log({"loss": loss.item()})

       # For logging, we accumulate the loss scaled by GRAD_ACCUM_STEPS to reflect the effective batch size.   
        running_loss += loss.item() * GRAD_ACCUM_STEPS
        avg_loss = running_loss / step
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # ---- Validation & BLEU after each epoch ----
    model.eval()
    gen_texts = []
    ref_texts = []
    val_loss_running = 0.0

    with torch.no_grad():
        pbar_val = tqdm(valid_loader, desc=f"Epoch {epoch} [valid]")
        for batch in pbar_val:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            
            outputs = model(**batch)
            val_loss_running += outputs.loss.item()

            # Generate translation using beam search with NUM_BEAMS
            # Check different options for sampling during generation in https://huggingface.co/docs/transformers/en/main_classes/text_generation
            generated_tokens = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
            )

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            labels = batch["labels"].clone()
            # Replace back id -100 with the PAD token for evaluation
            labels[labels == -100] = tokenizer.pad_token_id
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            gen_texts.extend([p.strip() for p in decoded_preds])
            ref_texts.extend([r.strip() for r in decoded_labels])

    bleu = corpus_bleu(gen_texts, [ref_texts])
    mean_val_loss = val_loss_running / max(1, len(valid_loader))

    print(f"Epoch {epoch}: train_loss={avg_loss:.4f} | valid_loss={mean_val_loss:.4f} | BLEU={bleu.score:.2f}")
    train_loss_history.append(avg_loss)
    valid_loss_history.append(mean_val_loss)
    bleu_history.append(bleu.score)

    np.save(OUT_DIR / "train_loss_history.npy", np.array(train_loss_history))
    np.save(OUT_DIR / "valid_loss_history.npy", np.array(valid_loss_history))
    np.save(OUT_DIR / "bleu_history.npy", np.array(bleu_history))

    if bleu.score > best_valid_bleu:
        best_valid_bleu = bleu.score
        save_dir = f"t5-{SRC_LANG[:2].lower()}2{TGT_LANG[:2].lower()}-best"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved new best model to: {save_dir}")
    
    