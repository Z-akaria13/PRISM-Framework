# Image shortcut (PyTorch)

A small CNN that labels grey scans as healthy or disease, with two problems planted for PRISM to find:
a scanner mark that the model uses as a shortcut for disease, and a disease sign that is fainter in
women's scans.

```bash
python make_data.py      # about 10 s: writes images/, metadata.csv and model.pt
prism
```

| Tab | Set |
|---|---|
| Setup | Project root: this folder · Connector: `connector.py` · Framework: PyTorch |
| Evidence | CSV: `metadata.csv` · ID: `scan_id` · Ground truth: `diagnosis` · Group By: `patient_id` |
| Wiretap | Tick `conv1`, `conv2` and `fc1` |
| Execution | Any output folder |

What you should see:

- **Performance**, sliced by `site`: the model misses more disease at site A and calls more disease
  at site B. Sliced by `sex`: it misses more disease in women.
- **Probing**: `site` can be read from `conv2` and `fc1` almost perfectly.
- **Error Gallery**: the missed cases, each next to its nearest neighbours in `fc1`; site B scans
  carry the scanner's bright mark in the top-left corner.

Or run every analysis at once: `prism audit <output folder> --slice site sex --concept site sex`.

`connector.py` is a starting point for a folder of images plus a CSV: copy it and change the lines
marked `CHANGE`.
