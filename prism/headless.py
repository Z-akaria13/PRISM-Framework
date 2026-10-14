"""
Headless extraction worker. Runs one job file (an AuditConfig written by the
GUI's cluster dispatch) on a machine without a display, e.g. a GPU node.

    python -u prism/headless.py --config prism_job_<id>.json     (or: prism run --config ...)

Prints the engine log to stdout, plus "[PROGRESS] <percent>" lines that the GUI's
monitor turns into progress. Exits non-zero if the extraction fails.
"""
import argparse
import os
import sys

if __name__ == "__main__" and not __package__:
    # Run as a file (python prism/headless.py): import PRISM as the `prism` package from the folder above,
    # never this folder's modules as loose top-level names (core, gui, ...), which would load a second copy.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from prism.core.config import AuditConfig
from prism.core.execution import run_extraction
from prism.core.metadata import read_metadata
from prism.core.subject import load_subject


def _emit(message):
    print(message, flush=True)  # the GUI tails our stdout from a file; don't let it sit in a buffer


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prism run", description="Run one extraction job file.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        config = AuditConfig.from_json(f.read())

    subject = load_subject(config.project_root, config.script_path, config.framework)
    metadata = read_metadata(config.csv_path, config.id_col)
    ok = run_extraction(config, subject, metadata,
                        progress_callback=lambda pct: _emit(f"[PROGRESS] {pct:.1f}"),
                        log_callback=_emit)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
