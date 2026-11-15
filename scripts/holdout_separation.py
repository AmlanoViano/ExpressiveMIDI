"""How well does a holdout model *rank* the unseen generator above real music?

    python scripts/holdout_separation.py models/audio_holdout_udio.pt

Prints AUC (held-out generator vs real test tracks), detection rate at the default 0.5
threshold, and detection rate at the threshold giving 1% false alarms on real test tracks.
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from expressivemidi.detect import audio  # noqa: E402

path = sys.argv[1]
cache = sys.argv[2] if len(sys.argv) > 2 else "data/audio_cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(path, map_location=device, weights_only=False)
hold = ck["holdout"]
model = audio.AudioDetector(ck.get("width", 32)).to(device)
model.load_state_dict(ck["state_dict"])
index = json.load(open(os.path.join(cache, "index.json")))
_, _, te = audio._split_items(index, hold)
items = [it for it in te if it["source"] in ("real/fma", f"ai/{hold}")]
dl = torch.utils.data.DataLoader(audio.ExcerptDataset(cache, items, False), batch_size=64, num_workers=0)
s = audio._track_scores(model, dl, len(items), device)
y = np.array([it["label"] for it in items])
thr1 = np.quantile(s[y == 0], 0.99)
out = dict(holdout=hold, n_heldout=int(y.sum()), n_real=int((y == 0).sum()),
           auc_vs_real=float(roc_auc_score(y, s)),
           detect_at_0_5=float(np.mean(s[y == 1] > 0)),
           detect_at_1pct_false_alarm=float(np.mean(s[y == 1] > thr1)),
           median_p_ai_heldout=float(np.median(1 / (1 + np.exp(-s[y == 1])))))
print(json.dumps(out, indent=1))
