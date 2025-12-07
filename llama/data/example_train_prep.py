# llama/data/example_train_prep.py
import pandas as pd
import json
import random
from pathlib import Path

def create_jsonl_from_coco(metadata_csv, out_train, out_val, sample_fraction=0.01, seed=42):
    df = pd.read_csv(metadata_csv)
    random.seed(seed)
    # group captions by image_id
    grouped = df.groupby("image_id")["caption"].apply(list).to_dict()
    # build simple examples: pick N random captions as context, pick one as target
    items = []
    ids = list(grouped.keys())
    for img_id in ids:
        caps = grouped[img_id]
        # sample some other captions to make context (simulate retrieved neighbors)
        neighbors = []
        for _ in range(3):
            other = grouped[random.choice(ids)]
            neighbors.append(random.choice(other))
        context = " || ".join(neighbors + [random.choice(caps)])
        target = random.choice(caps)
        items.append({"context": context, "target": target})

    random.shuffle(items)
    n = len(items)
    n_train = int(n * (1 - sample_fraction))
    train = items[:n_train]
    val = items[n_train:]

    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    Path(out_val).parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w", encoding="utf-8") as fh:
        for ex in train:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_val, "w", encoding="utf-8") as fh:
        for ex in val:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("Wrote", out_train, "and", out_val)

if __name__ == "__main__":
    create_jsonl_from_coco("data/coco/metadata_train.csv", "llama/data/train.jsonl", "llama/data/val.jsonl", sample_fraction=0.01)
