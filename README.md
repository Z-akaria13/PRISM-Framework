# PRISM

**Probing and Reporting Identity-based Shortcut Metrics.** PRISM audits a trained model for
demographic bias and shortcut learning. You point it at your model and a labelled test set; it
runs the model once, stores the hidden-layer activations, and then answers three questions:

- **Does the model work equally well for everyone?** Performance per subgroup (sex, site, age
  band…), with the gaps between groups and a 95% interval on each.
- **What has the model learned to see?** Linear probes test whether an attribute such as sex or
  scanner can be read off a hidden layer, against shuffled-label controls.
- **What do its mistakes look like?** An error gallery shows each false positive and false
  negative next to its nearest neighbours in the model's own representation.

PRISM began as a tool for auditing MRI-based Alzheimer's classifiers (MSc thesis, Warwick DCS)
and works with any PyTorch or scikit-learn model. Keras models are supported for their outputs;
extracting their hidden layers is a known open bug (`debt.MD`, P0 #5).

## Install

PRISM needs Python 3.10 or newer, and Tkinter for its window.

```bash
git clone <repo-url> prism
cd prism
pip install -e ".[torch]"     # PyTorch models
pip install -e .              # scikit-learn models only (no PyTorch needed)
pip install -e ".[keras]"     # Keras / TensorFlow models
```

This installs the `prism` command. Tkinter ships with most Python installations; if yours lacks it:

| Platform | Command |
|---|---|
| **Ubuntu/Debian** | `sudo apt-get install python3-tk` |
| **Fedora/RHEL** | `sudo dnf install python3-tkinter` |
| **Arch** | `sudo pacman -S tk` |
| **macOS (Homebrew)** | `brew install python-tk@3.xx` (match your Python version) |
| **Windows** | Re-run the Python installer → check **tcl/tk and IDLE** |

## Connect your model

PRISM reaches your model through a *connector*: a short Python file in your own project that
says how to load the model, how to load one item, and how to run a batch.

```python
import torch
from prism import PrismConnector

class MyConnector(PrismConnector):
    class_names = ["CN", "AD"]   # optional: the label of each output column, in order

    def load_model(self):
        model = MyNet()   # your own model class
        model.load_state_dict(torch.load(f"{self.project_root}/model.pth", map_location="cpu"))
        return model

    def load_item(self, row_id, metadata_row):
        # row_id is this row's value in the ID column you choose (e.g. a file name), as text;
        # load_scan stands for however you already read one input
        return load_scan(f"{self.project_root}/scans/{row_id}.nii.gz")

    def run_inference(self, model, batch_data):
        return model(torch.stack(batch_data))
```

PRISM handles the rest: it puts PyTorch models in evaluation mode, moves them to the GPU when
there is one, batches the rows, and stores every result next to its row ID. Items that fail to
load, or batches the model crashes on, are left out and listed, with the error, next to the
results; every analysis says how many are missing and which groups lost the most. If more than 5%
of all rows fail (the Execution tab's **Stop if more than** setting), the run stops instead, since
failures often share a cause, such as one site's files, and would remove a whole group. [`prism/connector.py`](prism/connector.py) documents the whole protocol in one file,
including the optional `output_kind` and `render_item_for_gallery`. It is also a good file to hand
to a colleague or an AI assistant along with your model code, if someone else writes the connector.

## Use the GUI

```bash
prism
```

**File → Save Setup** saves everything the tabs hold, cluster settings included, and **File → Open
Setup** fills them in again, so a setup is entered once. A saved setup is also a job file:
`prism run --config <setup file>` runs it without the window.

Work through the seven tabs:

1. **Setup**: choose your project folder, connector script and framework, then **INGEST SUBJECT**.
2. **Evidence**: load the metadata CSV and choose the ID, target and Group By (e.g. patient ID)
   columns. Every other column is screened as a possible proxy for the target.
3. **Wiretap**: tick the hidden layers to record, and right-click selected rows to set how each is
   pooled. **Show shapes** runs one row through the model and shows each layer's output shape and
   how many values its pooling keeps; the info button lists what each pooling keeps.
   Empty for scikit-learn models, which have no layers to record.
4. **Execution**: run the model over every row, here or as a cluster job, with progress and Cancel.
5. **Performance**: metrics and fairness gaps per subgroup, for binary, multiclass or regression
   targets. Each figure has a 95% bootstrap interval that resamples whole subjects when Group By is
   set; groups below Min Group Size are shown but not compared.
6. **Probing**: can a metadata attribute be read off a layer? Scored against shuffled-label controls,
   with a permutation p-value.
7. **Error Gallery**: browse misclassifications and their nearest neighbours in a layer's space.

## Write a report

Once an extraction has finished, one command runs every analysis on its results and writes a
Markdown report (`prism_report.md` in the results folder) that leads with the findings:

```bash
prism audit <results dir>                          # target and Group By as recorded with the run
prism audit <results dir> --slice sex site --concept sex site age --json audit.json
prism audit --help                                 # every option
```

It covers the confound screen, performance and fairness gaps for each slice column, and a linear
probe of every layer for every concept. Probing many layers for many concepts turns some chance
results into apparent findings, so the probes' p-values are corrected across the whole report
(Benjamini–Hochberg, false discovery rate 0.05 by default, `--fdr`), and each probe gets enough
shuffled-label controls for the correction to be able to pass it. It exits 2 if the options or the
results cannot be used.

## Run on a cluster

In the Execution tab, choose **Dispatch to Cluster** and fill in **Cluster Settings** (an SSH
prefix if you're not on the login node, the job's resource headers, and environment setup).
PRISM writes a job file and a batch script into the output folder, submits it with `sbatch`, and
follows the job's log, showing progress and reporting a job the scheduler kills. The compute node
needs the same paths as your machine (a shared filesystem). On the node the job runs PRISM's job
runner, the same as `prism run --config <job file>`, which you can also run yourself.

The cluster path is tested end to end against a local stand-in for SLURM, not on a real cluster.

## Results files

Every extraction writes these files to its output folder:

- `prism_embeddings.h5`: the model's outputs, one dataset per recorded layer, every row's ID, and
  the run's settings (columns, layers, pooling, device) and class order, and what produced them:
  PRISM's and the packages' versions, when it ran, and fingerprints of the metadata CSV, the
  connector and the model's weights. The audit report ends with these.
- `prism_metadata.csv`: the metadata rows that were extracted, in the same order.
- `prism_failed_rows.csv`, only if some rows failed: their metadata, where each one failed and the
  error.

When the results are opened, PRISM checks that the files still match row for row. A new run in a
folder that already holds results replaces them only once it has finished; until then, and if it
fails or is cancelled, the earlier results stay as they were. Results written by
earlier versions (`whydunnit_*` files) still open. See [`architecture.MD`](architecture.MD) for the
full format.

## Add support for another framework

A framework is supported by an *inspector*, a subclass of `prism.inspectors.base.Inspector` that
lists a model's layers and records their outputs during a batch. Another package can add one
through an entry point in its `pyproject.toml`; it then appears in the Setup tab's framework list:

```toml
[project.entry-points."prism.inspectors"]
"JAX" = "my_package.jax_inspector:JaxInspector"
```

## Examples

Two examples in `examples/` generate their own data and train their own model in seconds
(`python make_data.py`), with problems planted for PRISM to find. Each folder's README says how to
fill in the tabs and what you should see. They need `pip install -e ".[examples]"`.

- [`image_shortcut`](examples/image_shortcut) (PyTorch): a small CNN on grey scans that uses a
  scanner mark as a shortcut for disease, and misses more disease in women.
- [`tabular_bias`](examples/tabular_bias) (scikit-learn): a loan model trained on records biased
  against one region, which it learns through a proxy, and a column that leaks the outcome.

Their connectors are starting points for your own: a folder of images plus a CSV, and a pickled
scikit-learn model with its inputs in the CSV. Copy one and change the lines marked `CHANGE`.

## Running the tests

```bash
pip install -r requirements-dev.txt    # the same as: pip install -e ".[torch,examples,dev]"
pytest                                 # full suite (~3 min)
pytest -m "not slow and not gui"       # skip the slow and GUI tests (~1 min)
pytest --cov=prism                     # with coverage
pytest -m scale -s                     # opt-in: 100,000 rows through extraction, a probe and the intervals
```

GUI tests build the real window hidden and skip themselves if no display is available. The cluster
test really runs the generated batch script (it needs bash: Git Bash on Windows, or set
`PRISM_TEST_BASH`). Tests marked `known_bug` are strict xfails that pin open items in `debt.MD`;
when a fix makes one pass, pytest says so, so the marker can be removed.

## Further reading

- [`architecture.MD`](architecture.MD): how the pieces fit together.
- [`debt.MD`](debt.MD): known defects and their fixes.

## Licence

PRISM is free software: you can redistribute it and modify it under the terms of the GNU General
Public License, version 3 or (at your option) any later version. See [`LICENSE`](LICENSE).
Copyright (C) 2026 Zakaria.

## Citing PRISM

If you use PRISM in your work, please cite it. [`CITATION.cff`](CITATION.cff) has the details, and
GitHub's "Cite this repository" button turns it into APA or BibTeX.
