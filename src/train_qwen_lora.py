import argparse
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def build_inputs(record, processor, device):
    messages = record["messages"]

    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_text = processor.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    full_inputs["labels"] = labels

    return move_to_device(full_inputs, device)


@torch.no_grad()
def evaluate(model, processor, val_records, device, max_val_samples=None):
    model.eval()

    if max_val_samples is not None:
        val_records = val_records[:max_val_samples]

    losses = []
    for rec in tqdm(val_records, desc="val", leave=False):
        batch = build_inputs(rec, processor, device)
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))

    model.train()

    if not losses:
        return None

    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    if torch.cuda.is_available():
        print("[INFO] gpu:", torch.cuda.get_device_name(0))

    train_records = load_jsonl(args.train_jsonl)
    val_records = load_jsonl(args.val_jsonl)

    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_records = val_records[: args.max_val_samples]

    print("[DATA] train records:", len(train_records))
    print("[DATA] val records:", len(val_records))

    processor = AutoProcessor.from_pretrained(args.model_name)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_update_steps = math.ceil(len(train_records) * args.epochs / args.grad_accum)
    warmup_steps = int(total_update_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    print("[TRAIN] total_update_steps:", total_update_steps)
    print("[TRAIN] warmup_steps:", warmup_steps)

    best_val_loss = float("inf")
    global_step = 0

    history_path = out_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_records)

        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_records, desc=f"epoch {epoch}/{args.epochs}")

        for i, rec in enumerate(pbar, start=1):
            batch = build_inputs(rec, processor, device)

            out = model(**batch)
            loss = out.loss / args.grad_accum

            loss.backward()
            running += float(loss.detach().cpu()) * args.grad_accum

            if i % args.grad_accum == 0 or i == len(train_records):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix({
                "loss": running / i,
                "lr": scheduler.get_last_lr()[0],
                "step": global_step,
            })

        train_loss = running / max(len(train_records), 1)

        val_loss = evaluate(
            model,
            processor,
            val_records,
            device,
            max_val_samples=args.max_val_samples,
        )

        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
        }

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print("[EPOCH RESULT]", row)

        last_dir = out_dir / "last_lora"
        model.save_pretrained(last_dir)
        processor.save_pretrained(last_dir)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = out_dir / "best_lora"
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            print(f"[BEST] saved best_lora | val_loss={best_val_loss:.6f}")

        if args.save_every_epoch:
            ep_dir = out_dir / f"epoch_{epoch:03d}_lora"
            model.save_pretrained(ep_dir)
            processor.save_pretrained(ep_dir)

    print("\n[DONE]")
    print("best_val_loss:", best_val_loss)
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()
