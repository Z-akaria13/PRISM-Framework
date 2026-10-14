# PRISM — Architecture

> PRISM (Probing and Reporting Identity-based Shortcut Metrics), originally named
> "Whydunnit". A Tkinter desktop tool that audits a trained model for demographic
> bias and shortcut learning. It runs the model over a labelled dataset, stores the
> hidden-layer activations, and then runs fairness and probing analyses on them.

~5,600 lines of Python across 35 files, plus a pytest suite in `tests/`. Packaged by `pyproject.toml` as `prism-audit`: installing it adds the `prism` command (`python -m prism` does the same). Its dependencies are what PRISM itself imports; a model's framework (PyTorch, TensorFlow) is an extra.

---

## 1. High-level shape

```
                ┌──────────────────────── user code ─────────────────────────┐
                │  connector.py  (subclass of PrismConnector)                 │
                │    load_model() · load_item(row_id, meta) · run_inference() │
                │    render_item_for_gallery()                                │
                └───────────────▲─────────────────────────────────────────────┘
                                │ core/subject.py load_subject()  (via core/loader.py)
┌───────────────────────────────┴──────────────────────────────────────────────┐
│ gui/app.py  PrismApp(tk.Tk)  ── global mutable state "bag" ──                │
│   connector_instance · model_subject · inspector · project_root · script_path│
│   framework · metadata_df · csv_path · config_*_col · active_cache_dir       │
│                                                                              │
│  Tabs (gui/frames/*):                                                        │
│  1 Setup ─► 2 Evidence ─► 3 Wiretap ─► 4 Execution ─► 5 Performance          │
│                                            │        ├► 6 Probing             │
│                                            │        └► 7 Error Gallery       │
└────────────────────────────────────────────┼─────────────────────────────────┘
                     Execution tab builds an │ AuditConfig (core/config.py)
                       ┌─────────────────────┴──────────────────────┐
                 Run Locally                                Dispatch to Cluster
                       │                              core/dispatch.py: job.json + run.sh
                       │                              → sbatch → headless.py on the node
                       ▼                                            ▼
          core/execution.py run_extraction(config, subject, metadata, …)   ◄── same code both ways
                       │       ▲
                       │       └── inspectors/ (PyTorch · Keras · Sklearn): to_device, attach_hooks,
                       ▼                                                    retrieve_features
                         <output_dir>/prism_embeddings.h5       (one dataset per layer
                                                                  + "__model_output__")
                         <output_dir>/prism_metadata.csv        (rows that survived)
                                               ▲
             Tabs 5/6/7 read this cache back ──┘

  Side channel:
   analysis.py   – the Tk-free analysis behind Evidence/Performance/Probing,
                   also usable directly from scripts
```

## 2. Modules

### Entry points and the connector
| File | Role |
|---|---|
| `prism/cli.py`, `prism/__main__.py` | The `prism` command (`[project.scripts]` in `pyproject.toml`) and `python -m prism`: no command or `gui` opens the window, `audit` runs `audit.py`, `run` runs `headless.py`, `--version` prints `prism.__version__`. Each command imports only what it needs. |
| `prism/connector.py` | `PrismConnector` ABC, the user protocol: `load_model`, `load_item(row_id, meta)` (where `row_id` is the row's ID-column value as a string), `run_inference`, the optional `render_item_for_gallery` (default: the item as text; `image`, `text` and `table` payloads are drawn), and the optional `class_names` and `output_kind` attributes. Exported as `prism.PrismConnector`; `WhydunnitConnector` is the same class under its old name. |
| `prism/main.py` | Starts the GUI (`PrismApp`). Run as a file (`python prism/main.py`) it imports PRISM from the folder above, so every import is `prism.*` and no module is loaded twice under a second name. |
| `prism/audit.py` | `prism audit <results dir>` (or `python prism/audit.py`): `run_audit` runs the confound screen, `fairness_audit` for each slice column and `linear_probe` for every layer x concept on one cache, then corrects the probes' p-values across the report (`benjamini_hochberg`), with enough controls per probe for the correction to pass a probe no shuffle matched (`controls_for_fdr`: 2 x probes / FDR, 50 to 1,000). `render_report` writes Markdown that leads with the findings; `--json` writes the raw results. Columns default to those recorded with the cache. Exits 2 on unusable options or caches. |
| `prism/headless.py` | Cluster worker (`prism run --config <job>`, or the file itself): `AuditConfig.from_json` → `load_subject` → `run_extraction`. Prints the log and `[PROGRESS] <pct>` lines with explicit flushing, and exits 1 if the extraction fails. |
| `prism/analysis.py` | Library functions with no Tk dependency: `confound_screen`, `fairness_audit`, `subgroup_performance` (the binary view from model scores, over `fairness_audit`), `linear_probe`, `benjamini_hochberg`, `controls_for_fdr`, `format_audit_report` (a compact summary; `audit.py` writes the full report). A probe's controls are scored together: one ranking per fold for all shuffled label sets. Every analysis tab calls these: Evidence `confound_screen`, Performance `fairness_audit`, Probing `linear_probe`. |

### `core/`
| File | Role |
|---|---|
| `connectorTemplate.py`, `connector_template.py` | Older import paths for the connector, kept so existing connectors load. Both re-export `prism.connector`'s class. |
| `loader.py` | `importlib` loads the user's script, adds `project_root` to `sys.path`, and picks the concrete `PrismConnector` subclass it defines: the most derived one when connectors build on each other, preferring classes defined in the script. Several unrelated connectors, an incomplete one, or none is an error that says which. Connectors that still import `core.connectorTemplate` (from when PRISM ran with its own folder on `sys.path`) get the real module through a small import hook; a `core` package of the user's own still wins. |
| `config.py` | `AuditConfig`: a frozen dataclass describing one job (paths, framework, ID/target/group columns, `layers: {name: pooling}`, batch size, output dir, device, and `max_failed_percent`, the share of rows that may fail before the run stops). Round-trips to the JSON job file, and still reads older job files that listed layers as `"name [Pooling]"` labels. |
| `metadata.py` | `read_metadata(path, id_col)`: the metadata CSV with the ID column as text (pandas would read `0042` as 42). Used by the Evidence tab, `headless.py` and `AuditCache.open`. |
| `subject.py` | `load_subject(root, script, framework) -> Subject(connector, model, inspector)`. Shared by the Setup tab and `headless.py`. |
| `cancel.py` | `Cancelled`: raised from a progress or check callback to stop long work. `run_extraction`, `linear_probe` and `fairness_audit` let it through after cleaning up. |
| `execution.py` | `run_extraction(config, subject, metadata, on_progress, on_log, check=None)`; `check` runs before each item is loaded, and `Cancelled` from it (or from `on_progress`) removes the hooks, deletes the half-written results and is re-raised. It moves the model to `config.device` via the inspector, batches rows, calls `load_item` with per-row fault tolerance, calls `run_inference` with per-batch fault tolerance, pools each layer's activations according to its strategy, and appends each batch through `CacheWriter` (all-or-nothing). Every row that fails is recorded with where (`load_item`, `run_inference`, `saving`) and why; once more than `max_failed_percent` of all rows have failed, or if no row was extracted, it stops with `[CRITICAL ERROR]` and saves nothing (`RunStopped`, logged without a traceback). It also checks that the ID column exists and is unique. It doesn't import torch or Tk. |
| `provenance.py` | What produced a set of results: `environment()` (PRISM, Python, platform, package versions), `file_digest` (SHA-256 of the CSV and connector), and `inputs()`, which adds the inspector's `model_fingerprint()` and is taken as a run starts. |
| `failures.py` | Describes failed rows in plain language for the log, the cache's warnings and the report: how many failed at each stage with the first error, and the groups (columns with 2 to 20 values, not the ID or Group By column) that lost every row or at least twice the overall share, at least 2 rows each. |
| `cache.py` | The results files. `CacheWriter` appends whole batches only: every dataset and the `__ids__` dataset grow together, or the batch is rejected with `BatchMismatch`. It writes `*.partial` files and moves them into place in `finish()`, so earlier results survive a run that fails or is cancelled. `AuditCache.open()` checks IDs and lengths against the CSV before anything is analysed, and reads older caches in a legacy mode. |
| `labels.py` | `class_order()` and `decode_predictions()`: one way of turning model outputs into labels, used by both Performance and Gallery. |
| `dispatch.py` | Writes the job file and the bash/SBATCH script (`python -u headless.py …`), and builds the submit command (`SchedulerCommand`): through SSH, ssh's own arguments with the remote command as the last, `cd <quoted dir> && sbatch …`, so no local shell reads it; without SSH, the submit command run by the local shell in the output directory. Paths are `shlex.quote`d for the POSIX shells that read them. `watch_job` follows a submitted job: it reads only the log's new bytes, relays lines and `[PROGRESS]`, and asks the scheduler (`status_cmd`, default `squeue -h -j {job_id}`) every 30 s whether the job still exists, so a job killed at its time or memory limit is reported as lost instead of watched forever. `cancel_job` runs `cancel_cmd` (default `scancel {job_id}`). No Tk. |

### `inspectors/` — framework plugins
`inspectors/__init__.py` is the registry: the three built-in plugins by display name and import path (`BUILT_IN`), plus any that another package declares in the `prism.inspectors` entry-point group (a plugin may not take a built-in's name). Each is an `InspectorRef`, imported the first time it is used, so PyTorch or TensorFlow is needed only for audits of models built with it; a missing framework is reported with its `pip install` line. A plugin subclasses `inspectors/base.py`'s `Inspector`:

```
__init__(model) · extracts_hidden_layers: bool · discover_topology() -> [{id,type,params}]
attach_hooks(names) · retrieve_features(expected_rows) -> {name: ndarray} · clear_buffer() · teardown()
model_fingerprint() -> 'sha256:…' | None   (the loaded weights, recorded with the results)
output_shapes(names, run) -> {name: shape}   (one item's output shapes, for Show shapes; the base class
                           captures activations, the PyTorch plugin records only shapes)
to_device(device) -> str   (optional; the engine calls it at run start if present)
prepare_for_inference() -> warning | None · inference_context()   (optional; PyTorch: eval mode, torch.no_grad)
```

| Plugin | Mechanism |
|---|---|
| `PyTorchInspector` | `to_device` moves the model to the run's device (`auto` = CUDA when available) when the run starts, not when the inspector is created. `attach_hooks` registers forward pre-hooks on the model (counting whole-model passes, and moving inputs to the model's device) plus `register_forward_hook` on the selected leaf modules, buffering `detach().cpu().numpy()` of each output tagged with its pass and its call within that pass. A module called k > 1 times per pass comes back as `name@0` … `name@<k-1>`. `teardown` removes all of these hooks. |
| `KerasInspector` | Builds a sub-model that outputs the selected layers and tries to intercept the forward pass by assigning `model.__call__` on the instance. That assignment never takes effect; see debt.MD. It also transposes NHWC to NCHW. |
| `SklearnInspector` | No-op. Black-box mode with only `__model_output__`. |

`gui/setup_file.py` backs File → Save Setup / Open Setup: `collect(app)` reads every tab into a dict that is an `AuditConfig` job file plus `run_mode` and `cluster`; `open_setup` fills Execution and Evidence at once, ingests the subject (`SetupFrame.ingest(..., then=...)`) and ticks the layers when it has loaded (`WiretapFrame.tick_layers`), and reports what it couldn't restore in one warning.

### `gui/frames/` — the seven tabs
| Tab | What it does | Reads / writes |
|---|---|---|
| Setup | Picks the root, script, and framework, then runs ingestion as a background `Task` (`load_subject` → `discover_topology`) with a busy bar and status line, and fills the Wiretap tree on the Tk thread. A failed ingest is shown in an error dialog | writes `app.connector_instance`, `model_subject`, `inspector`, `project_root`, `script_path`, `framework` |
| Evidence | Loads the CSV, maps the ID/target/group columns, and screens every column as a possible proxy for the target (`analysis.confound_screen`: out-of-fold AUC for categories, kept group-aware when Group By is set, and value AUC for continuous columns). The ID and Group By columns are not screened | writes `app.metadata_df`, `csv_path`, `config_*_col` |
| Wiretap | Treeview of layers. An Extract box (click it, or select rows and press Space) decides what is extracted; the selection only says which rows a right-click's pooling applies to, so layers can get different poolings without losing the ticks. Show shapes runs the first metadata row through the model in a background `Task` (`core/execution.preview_shapes`, refused while an extraction runs) and fills Output (one row's shape) and Stored (what the pooling keeps, `core/config.stored_shape`, updated when the pooling changes) | exposes `extracted_layers() -> {name: pooling}` |
| Execution | `build_config()` turns the tabs' state into an `AuditConfig`, and asks before replacing results already in the output directory. It then runs `run_extraction` locally as a background `Task`, or uses `core/dispatch.py` to write the job, submit it with `ssh … sbatch` and follow it with `watch_job`. **Cancel** stops a local run after the current item (nothing is saved), or stops watching a cluster job and, if the user says so, cancels it with `cancel_cmd`. Status and cancel commands are set in Cluster Settings | writes `app.active_cache_dir` |
| Performance | Opens the cache (`gui/utils.open_cache`) and calls `analysis.fairness_audit`. Treats the task as regression if the target is numeric with more than 10 unique values (`REGRESSION_MIN_UNIQUE`). Binary: confusion counts, TPR/TNR/FPR/FNR/precision/F1/selection per slice, with equal-opportunity, predictive-equality, demographic-parity and accuracy gaps. Multiclass: accuracy, balanced accuracy, per-class recall and selection rate per slice, with their gaps. Regression: RMSE, MAE, mean error (bias) and R² per slice, with RMSE/MAE/bias gaps. Every figure has a 95% bootstrap interval (1,000 resamples, of whole subjects when Group By is set). Groups below **Min Group Size** (default 20 subjects, or scans without Group By) are shown but kept out of the gaps; a rate gap above 0.15 is a `[WARNING]` only if its interval excludes 0, otherwise a "may be noise" `[NOTE]`. The ID and Group By columns are not offered as slices. Lists caveats (legacy cache, undeclared class order, rows without a slice value) as `[NOTE]`s under the report | reads `AuditCache` |
| Probing | Calls `analysis.linear_probe` with a **Probe Model** choice: shrinkage LDA (default) or a ridge classifier, both streamed from disk through `AuditCache.open_layer` (exact, memory set by features² not rows, controls nearly free), or multinomial logistic regression (loads the layer). All are scored by AUC and balanced accuracy; numeric concepts use streamed Ridge (R²), scaler fitted inside each fold, `StratifiedGroupKFold` when Group By is set, plus optional shuffled-label controls (Shuffles box: 1–1,000, default 50; lowest reachable p = 1/(shuffles+1)) with a permutation p-value. The streamed models handle layers up to about 10,000 features per row, any number of rows; LDA's many-shuffle case factorises each fold once (Woodbury updates), and held-out scores are kept in a fixed memory budget. Warns when Group By is unset | reads `AuditCache` |
| Error Gallery | Binary-only list of FP/FN. On click it calls the connector to load and render the item, then finds the 3 nearest neighbours in the chosen layer's space (Euclidean, computed directly on the layer it keeps in memory between clicks), skipping the suspect's own Group By group. It passes the connector the same `row_id` it got during extraction | reads `AuditCache` and calls the connector |

## 3. Results-file contract (`core/cache.py`)

The cache is the only thing that connects extraction to analysis.

```
prism_embeddings.h5
  attrs   schema_version=2 · audit_config (AuditConfig JSON: ID/target/group columns, layers, device …)
          class_names (JSON list, if the connector declares it) · output_kind · complete
          rows_attempted · rows_failed · provenance (JSON: versions, platform, start and end, input fingerprints)
  __ids__            (N,) str   every row's ID-column value, in row order
  __model_output__   (N, …)     whatever run_inference returned
  <layer name>       (N, …)     pooled activations; attrs "pooling", "output_shape" (one row's, before pooling)
  <module>@<i>       (N, …)     call i of a module run several times per pass; attrs "module", "call"
prism_metadata.csv              the N metadata rows, same order
prism_failed_rows.csv           the rows that failed, with prism_failed_at and prism_error (only if any did)
```

A run writes `<name>.partial` files and moves them into place when it finishes: the HDF5 file first, so if the run is interrupted between the moves its IDs disagree with the old CSV and opening refuses the pair. A finished run removes an earlier run's `prism_failed_rows.csv` when it has none of its own.

Results written before the rename (`whydunnit_embeddings.h5` / `whydunnit_metadata.csv`) open the same way, and a new run in that folder removes them.

* **Alignment is enforced.** Batches are appended all-or-nothing. A batch whose arrays disagree on row count, drop or add a layer, or change feature shape is skipped whole and logged. `AuditCache.open()` re-checks that the CSV's ID column matches `__ids__` (IDs are read as text everywhere, so `0042` stays `0042` from the Evidence tab to the connector to the results), and that every dataset has N rows.
* **Rows left out are counted and described.** `AuditCache.open()` reads `rows_failed` and the failed rows, and puts a plain-language note first in `warnings` (also `failure_note`): how many rows are missing, where they failed, and the groups most affected. Every analysis tab shows it, the Execution tab shows it in a dialog when a run finishes, and the audit report leads with it. Caches written before schema 2 have no record of failed rows.
* **The run's settings travel with the results.** If the Evidence tab wasn't confirmed in this session, the analysis tabs use the ID, target and Group By recorded in `audit_config` (`gui/utils.column_mapping`). Reopening a cache later keeps its grouping.
* **Class order is declared, or flagged.** Column *j* of a `(N,k)` output is `class_names[j]`. A single-column output scores `class_names[-1]` against `class_names[0]`, thresholded at 0.5 (or at 0 when `output_kind="logits"`). Without `class_names`, labels `0…k-1` map to themselves. Any other labels fall back to alphabetical order, with a `[NOTE]` in the report.
* **Legacy caches** (written before this format) open with a warning and are matched by position. Layers whose length disagrees with the CSV are left out, and misaligned predictions are refused.
* **Pooling** is still applied at write time; the strategy used is now recorded on each layer. Strategies: Mean Pool, Max Pool, Grid Pool (mean over a 4×4 / 4×4×4 grid), Seq Pool, CLS Token, Flatten.
* **Reused modules:** `AuditCache.layer_names` also lists a virtual `<module>@all`, read as every call side by side (`calls_of`).
* **Writing at scale:** datasets are chunked in ~1 MB blocks of whole rows; after the first batch the engine projects the final size and stops (removing the half-written file) if it would not fit on the disk.
* **Fairness intervals:** `fairness_audit` turns the predictions into counts per (bootstrap unit, slice) — a K×K confusion matrix, or regression sums — held as a sparse matrix, so each resample is one sparse product with a vector of resampling weights. Weights are drawn in blocks within a fixed memory budget. A gap is the best minus the worst eligible group, the pair fixed by the point estimate, and its interval is that pair's difference across the same resamples. 100,000 scans with 1,000 resamples take about 1 s by subject (25,000 subjects) and 5 s by scan. Resampling subjects matters when people have several scans: in a simulation with four agreeing scans per person and no true gap, the subject intervals contained 0 in 96% of 300 runs, the scan intervals in only 79%.
* **Reading:** `AuditCache.layer(name)` loads one layer; `AuditCache.open_layer(name)` hands out the on-disk dataset so streamed analyses (the ridge probe) read it in row chunks and never hold it whole.

## 4. Control and threading model

* Tk mainloop runs on the main thread, and only the main thread touches widgets.
* Slow work runs through `gui/worker.py`. A `Task` runs `work(task)` on a daemon thread; the work reports through `task.progress`, `task.log` and `task.status`, which only put messages on a queue, and the Tk thread drains that queue with `after()`, so every callback (`on_progress`, `on_log`, `on_status`, then one of `on_done` / `on_error` / `on_cancel`, then `on_finish`) runs on the Tk thread. `task.cancel()` makes the next `task.progress`, `task.check()` or `task.sleep()` in the work raise `Cancelled`.
* A `Runner` keeps one task at a time per tab: a new request (e.g. another Slice By choice or a click on another Gallery item) supersedes the current task, whose results are dropped, and starts once it has ended, so a user's connector is never called from two threads at once.
* Background work: ingestion (Setup), extraction and the cluster watch (Execution, with Cancel), probes (Probing, with a progress bar and Cancel), the bootstrap (Performance) and item loads plus the neighbour search (Gallery). The Evidence tab's confound screen still runs on the Tk thread.
* Figures are `matplotlib.figure.Figure` objects, not pyplot ones, so redrawing does not accumulate figures.
* State is shared through attributes on the `tk.Tk` root object. They are initialised in `PrismApp.__init__`, and each tab writes its own (on the Tk thread). Execution reads them, plus `frame_wiretap.selected_layers()`, to build the `AuditConfig`. The extraction engine never sees the GUI.

## 5. Cluster dispatch flow

```
Execution tab: build_config() ──► AuditConfig
core/dispatch.py ──► outdir/prism_job_<id>.json   (AuditConfig.to_json())
                 ──► outdir/prism_run_<id>.sh     (SBATCH headers + env + python -u headless.py)
Execution tab    ──► SchedulerCommand.run(): ssh … "cd <outdir> && sbatch run.sh"  (argv, no local shell)
                     or "sbatch run.sh" in <outdir> by the local shell              (in a Task)
core/dispatch.watch_job ──► tail outdir/prism_log_<id>.txt (new bytes only): log lines → console,
                     [PROGRESS] → bar; "[SUCCESS]" = complete, "[CRITICAL ERROR]" = failed (lines written
                     just after the verdict are still shown); every 30 s "[ssh …] squeue -h -j <job id>":
                     if the job has left the queue and the log has no verdict 30 s later → lost
compute node     ──► headless.py: AuditConfig.from_json → load_subject → run_extraction (device "auto" = node's GPU)
```
This needs a filesystem shared between the GUI host and the compute node, and the same absolute paths on both. The GUI still loads the model locally so Wiretap can list layers, but on the CPU. The GPU is claimed only by the machine running the extraction.

## 6. External dependencies

* Required: numpy, pandas, scikit-learn, joblib (installed with scikit-learn; it fingerprints scikit-learn models), scipy, h5py, matplotlib, and Tkinter (part of most Python installations). `pyproject.toml` lists the same as `requirements.txt`, and a test keeps every import PRISM makes declared there.
* Extras: `torch` (the PyTorch plugin), `keras` (TensorFlow, for the Keras plugin), `examples` (torch, torchvision and pillow for the bundled examples), `dev` (pytest). torch and TensorFlow are imported only by their plugins, and the window's device label imports torch in the background only if it is installed, so a scikit-learn audit runs without either.

## 7. Examples (`examples/`)
Outside the package. Each has a `make_data.py` that generates its data and trains its model (seeded, seconds on a CPU) next to itself, a `connector.py` written as a starting point for users, a README, and a `.gitignore` for what it generates. `tests/test_examples.py` runs each end to end through `run_audit` and checks the planted findings; removing a planted bias fails its test.

| Example | Planted |
|---|---|
| `image_shortcut` (PyTorch) | 32×32 scans; a disease spot fainter in women's scans (TPR gap, M vs F); a corner mark on site B's scans, with site B mostly diseased in training only, so the model leans on it (TPR and selection gaps by site; `site` readable from `conv2` and `fc1`). Zero-padded `scan_id`s, two scans per patient. |
| `tabular_bias` (scikit-learn) | Outcomes depend only on income, credit score and debt ratio, but a third of Southern repayments are recorded as defaults in training; the model learns region through `branch` (selection and FPR gaps, South vs North). `account_status`, recorded after the outcome, leaks it (HIGH in the screen). |
