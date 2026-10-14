"""
THE prism COMMAND
    prism                                  open the GUI (also: prism gui, python -m prism)
    prism audit <results dir> [options]    write an audit report for one extraction (prism audit --help)
    prism run --config <job file>          run one extraction job, e.g. on a compute node
    prism --version
"""
import sys

from prism import __version__

USAGE = __doc__.split("\n", 2)[2].rstrip()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command, rest = (argv[0], argv[1:]) if argv else ("gui", [])
    if command in ("-h", "--help", "help"):
        print("usage:\n" + USAGE)
        return 0
    if command in ("-V", "--version"):
        print(f"prism {__version__}")
        return 0
    if command == "gui":
        from prism.main import main as gui
        gui()
        return 0
    if command == "audit":
        from prism.audit import main as audit
        return audit(rest)
    if command == "run":
        from prism.headless import main as run
        return run(rest)
    print(f"prism: unknown command '{command}'\n\nusage:\n{USAGE}", file=sys.stderr)
    return 2
