"""
Generate this example's scans and train its model (about 10 s on a CPU):

    python make_data.py

Writes, next to this file:
    images/<scan_id>.png   400 scans to audit (32×32, grey), 2 per patient
    metadata.csv           one row per scan: scan_id, patient_id, site, sex, age, diagnosis
    model.pt               the trained ScanNet's weights (model.py)

What is planted, for PRISM to find:
    - Disease shows as a faint bright spot. It is fainter in women's scans, so the model misses
      more of their disease.
    - Scanners at site B burn a small bright square into the corner of every scan. In the
      training data most site B patients have the disease, so the model learns the square as a
      shortcut: in the audit data, where both sites have as many patients with the disease, it
      over-calls disease at site B and misses it at site A.
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

from model import ScanNet

HERE = os.path.dirname(os.path.abspath(__file__))
SIZE = 32
SPOT = {"M": 0.36, "F": 0.20}          # brightness of the disease spot, by sex
TRAIN_DISEASE_RATE = {"A": 0.25, "B": 0.75}  # in the training data only; 0.5 at both sites when audited
SEED = 0


def scan(rng, disease, sex, site):
    """One 32×32 scan in [0, 1]."""
    image = rng.normal(0.3, 0.1, (SIZE, SIZE))
    if disease:
        y, x = rng.integers(8, SIZE - 8, 2)
        yy, xx = np.mgrid[:SIZE, :SIZE]
        image += SPOT[sex] * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5 ** 2))
    if site == "B":
        image[1:4, 1:4] = 0.9  # the scanner's burnt-in mark
    return np.round(np.clip(image, 0, 1) * 255) / 255  # as stored in an 8-bit PNG


def patients(rng, n, disease_rate):
    """n patients: site, sex, age and diagnosis; disease_rate maps site -> share with the disease."""
    site = rng.choice(["A", "B"], n)
    rate = np.array([disease_rate[s] for s in site])
    return pd.DataFrame({"patient_id": [f"P{i:03d}" for i in range(n)], "site": site,
                         "sex": rng.choice(["F", "M"], n), "age": rng.integers(55, 90, n),
                         "disease": rng.random(n) < rate})


def train(images, labels, rng):
    torch.manual_seed(SEED)
    model = ScanNet()
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    x = torch.tensor(images, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    for _ in range(8):
        for batch in np.array_split(rng.permutation(len(x)), len(x) // 64):
            optimiser.zero_grad()
            nn.functional.cross_entropy(model(x[batch]), y[batch]).backward()
            optimiser.step()
    return model.eval()


def main():
    rng = np.random.default_rng(SEED)

    training = patients(rng, 1500, TRAIN_DISEASE_RATE)
    images = np.stack([scan(rng, p.disease, p.sex, p.site) for p in training.itertuples()])
    model = train(images, training["disease"].to_numpy().astype(int), rng)
    torch.save(model.state_dict(), os.path.join(HERE, "model.pt"))

    audited = patients(rng, 200, {"A": 0.5, "B": 0.5})
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    rows = []
    for p in audited.itertuples():
        for _ in range(2):  # two scans per patient, e.g. two visits
            scan_id = f"{len(rows) + 1:04d}"
            pixels = scan(rng, p.disease, p.sex, p.site)
            Image.fromarray((pixels * 255).astype(np.uint8)).save(os.path.join(HERE, "images", f"{scan_id}.png"))
            rows.append({"scan_id": scan_id, "patient_id": p.patient_id, "site": p.site, "sex": p.sex,
                         "age": p.age, "diagnosis": "disease" if p.disease else "healthy"})
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "metadata.csv"), index=False)
    print(f"Wrote {len(rows)} scans, metadata.csv and model.pt to {HERE}")


if __name__ == "__main__":
    main()
