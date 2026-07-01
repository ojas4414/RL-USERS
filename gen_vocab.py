"""
gen_vocab.py — build the CORRECT vocab.json matching the training pipeline.
Replicates data/loader.py pipeline: filter_ → sort_split → build → dataset → build_vocab
Pure Python + json only (no pandas, no torch). Run once before simulate_live.py.
"""
import json
from collections import defaultdict

DATA        = "data/raw/All_Beauty.jsonl"
OUT         = "vocab.json"
WINDOW_SIZE = 3
MIN_COUNT   = 5

print("Reading reviews...")
rows = []
with open(DATA, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
        except Exception:
            continue
        uid = p.get("user_id")
        iid = p.get("parent_asin")   # loader.py reads parent_asin
        ts  = p.get("timestamp", 0)
        if uid and iid:
            rows.append((uid, iid, ts))
        if (i + 1) % 200_000 == 0:
            print(f"  {i+1:,} lines read...")

print(f"  {len(rows):,} valid rows")

# filter_: keep users and items that appear >= MIN_COUNT times
user_counts = defaultdict(int)
item_counts = defaultdict(int)
for uid, iid, _ in rows:
    user_counts[uid] += 1
    item_counts[iid] += 1

filtered = [
    (uid, iid, ts) for uid, iid, ts in rows
    if user_counts[uid] >= MIN_COUNT and item_counts[iid] >= MIN_COUNT
]
print(f"  {len(filtered):,} rows after filter (>={MIN_COUNT} interactions each)")

# sort_split: sort by timestamp, quantile(0.80) is train cutoff
filtered.sort(key=lambda r: r[2])
n            = len(filtered)
train_cutoff = filtered[int(n * 0.80)][2]

# build: sliding window pairs grouped by user
user_groups = defaultdict(list)
for uid, iid, ts in filtered:
    user_groups[uid].append((ts, iid))

pairs = []
for uid, events in user_groups.items():
    events.sort(key=lambda e: e[0])
    items      = [e[1] for e in events]
    timestamps = [e[0] for e in events]
    if len(items) < WINDOW_SIZE + 1:
        continue
    for i in range(len(items) - WINDOW_SIZE):
        pairs.append({
            "state":     items[i : i + WINDOW_SIZE],
            "action":    items[i + WINDOW_SIZE],
            "timestamp": timestamps[i + WINDOW_SIZE],
        })

print(f"  {len(pairs):,} session pairs")

# dataset: training split = timestamp <= train_cutoff
train_pairs = [p for p in pairs if p["timestamp"] <= train_cutoff]
print(f"  {len(train_pairs):,} training pairs")

# build_vocab: exactly as bc_trainer.build_vocab (idx starts at 0, no +1)
unique_items = set()
for p in train_pairs:
    unique_items.update(p["state"])
    unique_items.add(p["action"])

item_to_idx = {item: idx for idx, item in enumerate(sorted(unique_items))}
print(f"  {len(item_to_idx):,} unique products in training vocab")

with open(OUT, "w") as f:
    json.dump(item_to_idx, f)

print(f"Saved {OUT}  (vocab size: {len(item_to_idx):,})")
