import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd


def tokenize(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def modified_precision(pred_tokens_list, ref_tokens_list, n):
    clipped = 0
    total = 0

    for pred, ref in zip(pred_tokens_list, ref_tokens_list):
        pred_ngrams = Counter(ngrams(pred, n))
        ref_ngrams = Counter(ngrams(ref, n))

        total += sum(pred_ngrams.values())

        for ng, count in pred_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))

    if total == 0:
        return 0.0

    return clipped / total


def corpus_bleu(preds, refs, max_n=4):
    pred_tokens_list = [tokenize(x) for x in preds]
    ref_tokens_list = [tokenize(x) for x in refs]

    pred_len = sum(len(x) for x in pred_tokens_list)
    ref_len = sum(len(x) for x in ref_tokens_list)

    if pred_len == 0:
        return 0.0

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(pred_len, 1))

    precisions = []
    for n in range(1, max_n + 1):
        p = modified_precision(pred_tokens_list, ref_tokens_list, n)
        # smoothing for short medical sentences
        if p == 0:
            p = 1e-9
        precisions.append(p)

    score = bp * math.exp(sum(math.log(p) for p in precisions) / max_n)
    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--prediction_col", required=True)
    parser.add_argument("--reference_col", default="acute_target_sentence")
    parser.add_argument("--output_txt", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    preds = df[args.prediction_col].fillna("").astype(str).tolist()
    refs = df[args.reference_col].fillna("").astype(str).tolist()

    results = {
        "rows": len(df),
        "BLEU_1": corpus_bleu(preds, refs, max_n=1) * 100,
        "BLEU_2": corpus_bleu(preds, refs, max_n=2) * 100,
        "BLEU_3": corpus_bleu(preds, refs, max_n=3) * 100,
        "BLEU_4": corpus_bleu(preds, refs, max_n=4) * 100,
    }

    text = "\n".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in results.items()])
    print(text)

    if args.output_txt:
        Path(args.output_txt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_txt).write_text(text + "\n")
        print(f"\nsaved: {args.output_txt}")


if __name__ == "__main__":
    main()
