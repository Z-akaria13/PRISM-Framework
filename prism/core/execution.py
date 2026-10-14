import math
import numpy as np
import shutil
import traceback
from contextlib import nullcontext

from .cache import FAILED_NAME, MODEL_OUTPUT, BatchMismatch, CacheWriter, module_of, results_exist
from .cancel import Cancelled
from .config import DEFAULT_POOLING, GRID_CELLS_PER_AXIS, shape_text
from .failures import ERROR, FAILED_AT, describe, error_text, percent
from .provenance import environment, inputs, now


class RunStopped(Exception):
    """The run stopped for a reason the user can act on. Logged as `lines`, then the message, with
    no traceback."""

    def __init__(self, message, lines=()):
        super().__init__(message)
        self.lines = list(lines)


def _to_numpy(output):
    # Framework tensors (e.g. torch) expose detach(); checking for it avoids importing the framework here.
    if hasattr(output, "detach"):
        return output.detach().cpu().numpy()
    return np.array(output)


def _row_ids(df, id_col):
    """Every row's ID-column value as a string, checked to identify rows uniquely."""
    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' is not in the metadata.")
    ids = df[id_col].astype(str)
    repeated = sorted(set(ids[ids.duplicated()]))
    if repeated:
        shown = ", ".join(repeated[:5]) + (" ..." if len(repeated) > 5 else "")
        raise ValueError(f"ID column '{id_col}' must identify rows uniquely; repeated values: {shown}")
    return ids


def _grid_pool(data_matrix, cells=GRID_CELLS_PER_AXIS):
    """Average over a coarse grid of the spatial axes, as adaptive average pooling does:
    (batch, channels, *spatial) -> (batch, channels x cells^d). Keeps a rough layout of where
    activity is, at a fraction of Flatten's size."""
    batch, channels, spatial = data_matrix.shape[0], data_matrix.shape[1], data_matrix.shape[2:]
    bins = [[(int(np.floor(i * length / cells)), int(np.ceil((i + 1) * length / cells))) for i in range(cells)]
            for length in spatial]
    pooled = np.empty((batch, channels) + (cells,) * len(spatial), dtype=data_matrix.dtype)
    spatial_axes = tuple(range(2, 2 + len(spatial)))
    for cell in np.ndindex(*(cells,) * len(spatial)):
        region = tuple(slice(*bins[axis][i]) for axis, i in enumerate(cell))
        pooled[(slice(None), slice(None)) + cell] = data_matrix[(slice(None), slice(None)) + region].mean(axis=spatial_axes)
    return pooled.reshape(batch, -1)


def _prepare(layer_name, data_matrix, expected_rows, layer_strategies):
    """Reduce one captured activation to (rows, features) using its module's pooling strategy."""
    if layer_name == MODEL_OUTPUT:
        return data_matrix

    if data_matrix.shape[0] != expected_rows:
        raise BatchMismatch(f"'{layer_name}' produced {data_matrix.shape[0]} rows for {expected_rows} items. "
                            "Does the model run more than once per item (e.g. test-time augmentation)?")

    if len(data_matrix.shape) > 2:
        strategy = layer_strategies.get(module_of(layer_name), "Mean Pool")

        if strategy == "Flatten":
            data_matrix = data_matrix.reshape(data_matrix.shape[0], -1)

        # --- TRANSFORMER LOGIC ---
        elif strategy == "Seq Pool":
            # (Batch, Seq_Len, Embed_Dim) -> averages Seq_Len
            data_matrix = data_matrix.mean(axis=1)
        elif strategy == "CLS Token":
            # (Batch, Seq_Len, Embed_Dim) -> isolates the first token
            data_matrix = data_matrix[:, 0, :]

        # --- CNN LOGIC ---
        elif strategy == "Max Pool":
            # (Batch, Channels, Spatial) -> max of Spatial
            data_matrix = data_matrix.reshape(data_matrix.shape[0], data_matrix.shape[1], -1).max(axis=2)
        elif strategy == "Grid Pool":
            # (Batch, Channels, Spatial) -> mean over a coarse grid of Spatial
            data_matrix = _grid_pool(data_matrix)
        else:
            # Default: Spatial Mean Pool
            data_matrix = data_matrix.reshape(data_matrix.shape[0], data_matrix.shape[1], -1).mean(axis=2)
    return data_matrix


def _size(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}" if unit != "B" else f"{int(n_bytes)} B"
        n_bytes /= 1024


def _preflight(prepared, batch_rows, total_rows, output_dir, log_callback):
    """After the first batch: project the results' final size and stop before filling the disk."""
    per_row = sum(np.asarray(a).nbytes for a in prepared.values()) / batch_rows
    projected, free = per_row * total_rows, shutil.disk_usage(output_dir).free
    log_callback(f"Projected size of the results: {_size(projected)} for {total_rows} rows "
                 f"({_size(free)} free in {output_dir}).")
    if projected > free:
        raise ValueError(f"Not enough disk space: the results would need about {_size(projected)} but only "
                         f"{_size(free)} is free in {output_dir}. Use a pooling strategy (e.g. Grid Pool or "
                         "Mean Pool) instead of Flatten, fewer layers, or another disk.")
    if projected > free / 2:
        log_callback("[WARNING] The results will take more than half of the free disk space.")


def _note_reused_modules(features, log_callback):
    calls = {}
    for name in features:
        if module_of(name) != name:
            calls.setdefault(module_of(name), []).append(name)
    for module, names in calls.items():
        log_callback(f"[NOTE] '{module}' runs {len(names)} times in each forward pass, so each call is stored "
                     f"separately ({module}@0 ... {module}@{len(names) - 1}); analyses can use one call or "
                     f"{module}@all.")


def _note_shapes(features, prepared, layer_strategies, log_callback):
    """After the first batch: say what each layer's output looked like and what its pooling kept."""
    shapes = {}
    for name, data in features.items():
        if name == MODEL_OUTPUT:
            continue
        shapes[name] = tuple(np.shape(data)[1:])
        stored = tuple(np.shape(prepared[name])[1:])
        if len(shapes[name]) <= 1:
            log_callback(f"Layer '{name}': {shape_text(shapes[name])} values per row, stored as they are.")
        else:
            strategy = layer_strategies.get(module_of(name), DEFAULT_POOLING)
            log_callback(f"Layer '{name}': each row's output is {shape_text(shapes[name])}; {strategy} stores "
                         f"{shape_text(stored)}" + (" values." if len(stored) == 1 else "."))
    return shapes


def preview_shapes(subject, metadata, id_col, layer_names):
    """Run the first metadata row through the model and return {dataset name: shape of one row's
    output} for `layer_names` (a module run k times per pass gives name@0 ... name@<k-1>). The model
    is in eval mode for the run and left as it was."""
    connector, inspector = subject.connector, subject.inspector
    if not len(metadata):
        raise ValueError("The metadata has no rows.")
    row_id = _row_ids(metadata.iloc[:1], id_col).iloc[0]
    item = connector.load_item(row_id, metadata.iloc[0].to_dict())
    prepare = getattr(inspector, "prepare_for_inference", None)
    inference_context = getattr(inspector, "inference_context", nullcontext)

    def run():
        with inference_context():
            connector.run_inference(subject.model, [item])

    try:
        if prepare is not None:
            prepare()
        output_shapes = getattr(inspector, "output_shapes", None)
        if output_shapes is None:  # an inspector not built on Inspector
            from prism.inspectors.base import Inspector
            return Inspector.output_shapes(inspector, layer_names, run)
        return output_shapes(layer_names, run)
    finally:
        inspector.teardown()


def _failed_frame(df, failed):
    """The failed rows' metadata in row order, with where each one failed and why."""
    positions = sorted(failed)
    frame = df.iloc[positions].reset_index(drop=True)
    frame[FAILED_AT] = [failed[pos][0] for pos in positions]
    frame[ERROR] = [failed[pos][1] for pos in positions]
    return frame


def _describe_failed(df, failed, tried, exclude):
    """(failures by stage, most affected groups or None) among the first `tried` rows."""
    return describe(df.iloc[:tried], [pos in failed for pos in range(tried)], _failed_frame(df, failed), exclude)


def _unchanged(replacing):
    return " The results already in the output directory are unchanged." if replacing else ""


def run_extraction(config, subject, metadata, progress_callback, log_callback, check=None):
    """Run every metadata row through the model and cache outputs + selected layer activations.

    Writes <config.output_dir>/prism_embeddings.h5 and prism_metadata.csv (the rows that
    were extracted), and prism_failed_rows.csv for rows that failed (see core/cache.py). Results
    already there are replaced only when the run finishes. Problems are reported through
    log_callback; returns True on success.

    A row fails when load_item raises, and a batch's rows fail when run_inference raises or the
    outputs don't fit the other batches. Once more than config.max_failed_percent of all rows
    have failed, or if no row was extracted, the run stops and saves nothing.

    `check`, if given, is called before each item is loaded. Raising Cancelled from it (or from
    progress_callback) stops the run: the hooks are removed, the half-written results are
    deleted, and Cancelled is re-raised.
    """
    connector, inspector = subject.connector, subject.inspector
    writer = None
    replacing = False
    try:
        df = metadata
        batch_size = config.batch_size
        output_dir = config.output_dir
        total_rows = len(df)
        num_batches = math.ceil(total_rows / batch_size)
        row_ids = _row_ids(df, config.id_col)
        replacing = results_exist(output_dir)
        if replacing and not config.overwrite:
            raise ValueError(f"Results already exist in {output_dir}. Choose another output directory, "
                             "or allow replacing them.")
        exclude = [c for c in (config.id_col, config.group_col) if c]
        failed = {}  # row position -> (where it failed, error)
        max_failed = config.max_failed_percent / 100 * total_rows

        def record(positions, where, error):
            """Note rows that failed, and stop once more have failed than the limit allows."""
            text = error_text(error)
            for pos in positions:
                failed[pos] = (where, text)
            if len(failed) > max_failed:
                tried = max(positions) + 1
                stages, groups = _describe_failed(df, failed, tried, exclude)
                lines = [f"[WARNING] {len(failed)} of the {tried} rows tried so far failed: {stages}."]
                if groups:
                    lines.append(f"[WARNING] Most affected so far: {groups}.")
                raise RunStopped(
                    f"Stopped: more than {config.max_failed_percent:g}% of the {total_rows} rows failed, so nothing "
                    "was saved. Fix the connector or the data, or raise 'Stop if more than ... % of rows fail' "
                    "(max_failed_percent in a job file) to leave these rows out.", lines)

        layer_strategies = dict(config.layers)
        layer_names = list(layer_strategies)

        if not layer_names and getattr(inspector, 'extracts_hidden_layers', True):
            raise ValueError("No layers chosen: tick the layers to extract in the Wiretap tab.")
        started, run_inputs = now(), inputs(config, inspector, log_callback)  # before the run moves the model

        log_callback(f"Targeting layers: {layer_names}")

        # Inspectors that manage hardware place the model now, on whichever machine runs the job.
        to_device = getattr(inspector, "to_device", None)
        if to_device is not None:
            log_callback(f"Running on device: {to_device(config.device)}")
        prepare = getattr(inspector, "prepare_for_inference", None)
        if prepare is not None:
            warning = prepare()
            if warning:
                log_callback(f"[WARNING] {warning}")
        inference_context = getattr(inspector, "inference_context", nullcontext)

        inspector.attach_hooks(layer_names)

        writer = CacheWriter(output_dir, config,
                             class_names=getattr(connector, "class_names", None),
                             output_kind=getattr(connector, "output_kind", None))
        successful_positions = [] # TRACKER: Keep track of rows that survived

        log_callback("Starting fault-tolerant execution loop...")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_rows)

            batch_data = []
            batch_positions = []

            # A. Fault-Tolerant Data Loading
            for pos in range(start_idx, end_idx):
                if check is not None:
                    check()
                row_id = row_ids.iloc[pos]
                try:
                    item = connector.load_item(row_id, df.iloc[pos].to_dict())
                    batch_data.append(item)
                    batch_positions.append(pos)
                except Exception as e:
                    log_callback(f"[WARNING] Skipping row {row_id} (Load failed): {e}")
                    record([pos], "load_item", e)
                    continue

            if not batch_data:
                log_callback(f"[WARNING] Batch {batch_idx} is empty after filtering. Skipping.")
                continue

            # B. Fault-Tolerant Inference
            try:
                with inference_context():
                    raw_output = connector.run_inference(subject.model, batch_data)

                if raw_output is None:
                    raise ValueError("run_inference() returned None.")

                final_preds = _to_numpy(raw_output)

                features = inspector.retrieve_features(expected_rows=len(batch_data))
                inspector.clear_buffer()
                features[MODEL_OUTPUT] = final_preds

            except Exception as e:
                log_callback(f"[WARNING] Batch {batch_idx} inference crashed. Skipping batch: {e}")
                inspector.clear_buffer() # Clean up the mess
                record(batch_positions, "run_inference", e)
                continue # Skip saving this batch

            # C. Save to disk: the whole batch or none of it, so every dataset stays aligned with the CSV
            expected_rows = len(batch_data) # Use the filtered count!
            try:
                prepared = {name: _prepare(name, data, expected_rows, layer_strategies)
                            for name, data in features.items()}
                writer.append(row_ids.iloc[batch_positions], prepared)
            except BatchMismatch as e:
                log_callback(f"[ERROR] Batch {batch_idx} skipped, nothing from it was saved: {e}")
                record(batch_positions, "saving", e)
                continue

            if not successful_positions:
                _preflight(prepared, len(batch_positions), total_rows, output_dir, log_callback)
                _note_reused_modules(features, log_callback)
                writer.note_output_shapes(_note_shapes(features, prepared, layer_strategies, log_callback))

            # D. Record Success
            successful_positions.extend(batch_positions)
            progress_callback(((batch_idx + 1) / num_batches) * 100)

        if not successful_positions:
            lines = []
            if failed:
                stages, _ = _describe_failed(df, failed, total_rows, exclude)
                lines.append(f"[WARNING] {len(failed)} of {total_rows} rows failed: {stages}.")
            reason = f"all {total_rows} rows failed" if total_rows else "the metadata has no rows"
            raise RunStopped(f"No rows were extracted: {reason}, so there are no results to analyse.", lines)

        # 4. Cleanup & Save Filtered CSV
        inspector.teardown()

        # ONLY save the rows that successfully made it to the HDF5 file
        final_df = df.iloc[successful_positions].reset_index(drop=True)
        provenance = {**environment(), "started": started, "finished": now(), "inputs": run_inputs}
        writer.finish(final_df, _failed_frame(df, failed), rows_attempted=total_rows, provenance=provenance)

        if failed:
            stages, groups = _describe_failed(df, failed, total_rows, exclude)
            log_callback(f"[WARNING] {len(failed)} of {total_rows} rows ({percent(len(failed), total_rows)}) failed "
                         f"and are left out of the results: {stages}.")
            if groups:
                log_callback(f"[WARNING] Most affected: {groups}.")
            log_callback(f"[WARNING] {FAILED_NAME} lists each one with its error.")
        log_callback(f"[SUCCESS] Extracted {len(final_df)}/{total_rows} items to {output_dir}")
        return True

    except Cancelled:
        inspector.teardown()
        if writer is not None:
            writer.abort()
        log_callback("[CANCELLED] Extraction stopped; nothing was saved." + _unchanged(replacing))
        raise
    except RunStopped as e:
        inspector.teardown()
        if writer is not None:
            writer.abort()
        for line in e.lines:
            log_callback(line)
        log_callback(f"[CRITICAL ERROR] {e}" + _unchanged(replacing))
        return False
    except Exception as e:
        inspector.teardown()
        if writer is not None:
            writer.abort()
        log_callback(f"[CRITICAL ERROR] {str(e)}" + _unchanged(replacing))
        log_callback(traceback.format_exc())
        return False
