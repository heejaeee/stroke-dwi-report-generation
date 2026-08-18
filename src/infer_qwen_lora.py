import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def make_prompt_record(record):
    # Use only the user message, not the assistant target.
    return {
        "role": "user",
        "content": record["messages"][0]["content"],
    }


@torch.no_grad()
def generate_one(model, processor, record, device, max_new_tokens=128):
    messages = [make_prompt_record(record)]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    input_len = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_len:]

    output_text = processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_dir", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    if torch.cuda.is_available():
        print("[INFO] gpu:", torch.cuda.get_device_name(0))

    records = load_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    print("[DATA] records:", len(records))

    processor = AutoProcessor.from_pretrained(args.lora_dir)

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.to(device)
    model.eval()

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for rec in tqdm(records, desc="infer"):
            pred = generate_one(
                model=model,
                processor=processor,
                record=rec,
                device=device,
                max_new_tokens=args.max_new_tokens,
            )

            out = {
                "id": rec.get("id"),
                "patient_id": rec.get("patient_id"),
                "split": rec.get("split"),
                "source": rec.get("source"),
                "dataset_source": rec.get("dataset_source"),
                "original_split": rec.get("original_split"),
                "original_id": rec.get("original_id"),
                "prediction": pred,
                "target": rec.get("target"),
                "images": rec.get("images", []),
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print("[DONE]")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
