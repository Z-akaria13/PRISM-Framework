"""The `prism` command, the package metadata, and what an installed PRISM contains."""
import ast
import shutil
import subprocess
import sys
import zipfile

import pytest

import prism
from prism import cli
from _support import PRISM_DIR, REPO_ROOT

tomllib = pytest.importorskip("tomllib")  # Python 3.11+; PRISM itself runs on 3.10

PYPROJECT = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
# Import name -> distribution name, where they differ.
DISTRIBUTIONS = {"sklearn": "scikit-learn", "PIL": "pillow"}
# Framework plugins: the only modules that may import an optional framework at module level.
PLUGIN_EXTRAS = {"inspectors/pytorch.py": "torch", "inspectors/keras.py": "keras"}


def _name(requirement):
    """'scikit-learn>=1.3' -> 'scikit-learn'."""
    for i, char in enumerate(requirement):
        if char in "<>=!~[; ":
            return requirement[:i].strip().lower()
    return requirement.strip().lower()


def _prism_sources():
    for path in sorted(PRISM_DIR.rglob("*.py")):
        yield path.relative_to(PRISM_DIR).as_posix(), path


def _third_party_imports(path):
    """[(top-level module, imported at module level?)] for every absolute import in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []

    def visit(node, in_function):
        for child in ast.iter_child_nodes(node):
            nested = in_function or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                found.extend((alias.name.split(".")[0], not in_function) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.append((child.module.split(".")[0], not in_function))
            visit(child, nested)

    visit(tree, False)
    return [(module, top) for module, top in found
            if module not in sys.stdlib_module_names and module not in ("prism", "__future__")]


# ---------------------------------------------------------------------------
# The prism command
# ---------------------------------------------------------------------------

def test_version_matches_the_package(capsys):
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"prism {prism.__version__}"


def test_help_lists_every_command(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for usage in ("prism audit <results dir>", "prism run --config <job file>", "prism --version"):
        assert usage in out


def test_an_unknown_command_exits_2_with_the_usage(capsys):
    assert cli.main(["extract"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'extract'" in err
    assert "prism audit <results dir>" in err


@pytest.mark.parametrize("argv", [[], ["gui"]])
def test_no_command_or_gui_opens_the_window(monkeypatch, argv):
    import prism.main
    opened = []
    monkeypatch.setattr(prism.main, "main", lambda: opened.append(True))

    assert cli.main(argv) == 0
    assert opened == [True]


@pytest.mark.parametrize("command, module", [("audit", "prism.audit"), ("run", "prism.headless")])
def test_audit_and_run_get_the_rest_of_the_arguments(monkeypatch, command, module):
    import importlib
    target = importlib.import_module(module)
    calls = []
    monkeypatch.setattr(target, "main", lambda argv: calls.append(argv) or 7)

    assert cli.main([command, "results", "--json", "out.json"]) == 7
    assert calls == [["results", "--json", "out.json"]]


@pytest.mark.slow
def test_python_dash_m_prism_runs_the_command():
    def run(*args):
        return subprocess.run([sys.executable, "-m", "prism", *args], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=120)

    version = run("--version")
    assert (version.returncode, version.stdout.strip()) == (0, f"prism {prism.__version__}")
    audit_help = run("audit", "--help")
    assert audit_help.returncode == 0
    assert "usage: prism audit" in audit_help.stdout


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def test_the_console_script_and_version_point_at_real_code():
    assert PYPROJECT["project"]["scripts"] == {"prism": "prism.cli:main"}
    assert PYPROJECT["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "prism.__version__"}
    assert callable(cli.main)


def test_every_third_party_import_is_declared():
    core = {_name(r) for r in PYPROJECT["project"]["dependencies"]}
    extras = {extra: {_name(r) for r in reqs} for extra, reqs in PYPROJECT["project"]["optional-dependencies"].items()}
    problems = []
    for relative, path in _prism_sources():
        for module, top_level in _third_party_imports(path):
            dist = DISTRIBUTIONS.get(module, module).lower()
            if dist in core:
                continue
            extra = PLUGIN_EXTRAS.get(relative)
            if top_level and not (extra and dist in extras[extra]):
                problems.append(f"{relative} imports {module} at module level, but {dist} is not a core dependency")
            elif not any(dist in reqs for reqs in extras.values()):
                problems.append(f"{relative} imports {module}, which no extra provides")
    assert problems == []


def test_the_licence_and_citation_agree_with_the_package():
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    fields = dict(line.split(": ", 1) for line in citation.splitlines() if ": " in line and not line.startswith((" ", "-")))

    assert PYPROJECT["project"]["license"] == fields["license"] == "GPL-3.0-or-later"
    assert fields["version"] == prism.__version__ and fields["cff-version"] == "1.2.0"
    assert (REPO_ROOT / "LICENSE").read_text(encoding="utf-8").startswith(
        "                    GNU GENERAL PUBLIC LICENSE\n                       Version 3, 29 June 2007")


def test_requirements_txt_lists_the_core_dependencies():
    lines = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    listed = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    assert sorted(listed) == sorted(PYPROJECT["project"]["dependencies"])


@pytest.mark.slow
def test_prism_loads_without_pytorch_or_tensorflow():
    pytest.importorskip("tkinter")
    # A scikit-learn user's environment: importing either framework fails.
    code = ("import sys; sys.modules['torch'] = None; sys.modules['tensorflow'] = None\n"
            "import prism, prism.cli, prism.main, prism.audit, prism.headless, prism.analysis\n"
            "import prism.gui.app, prism.core.execution, prism.core.loader, prism.core.dispatch\n"
            "from prism.inspectors import get_inspectors\n"
            "get_inspectors()['Scikit-Learn'](object())\n"
            "print('ok')")

    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)

    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# The installed package
# ---------------------------------------------------------------------------

def _copy_source(destination):
    """The repository as a build sees it: pyproject, README, prism/ and examples/."""
    destination.mkdir()
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(REPO_ROOT / name, destination / name)
    for folder in ("prism", "examples"):
        shutil.copytree(REPO_ROOT / folder, destination / folder, ignore=shutil.ignore_patterns("__pycache__"))


@pytest.mark.slow
def test_the_wheel_holds_prism_and_its_command_but_not_the_examples(tmp_path):
    setuptools = pytest.importorskip("setuptools")
    if int(setuptools.__version__.split(".")[0]) < 77:
        pytest.skip("building needs setuptools 77 or newer")
    _copy_source(tmp_path / "src")

    build = subprocess.run([sys.executable, "-m", "pip", "wheel", str(tmp_path / "src"), "--no-deps",
                            "--no-build-isolation", "-w", str(tmp_path / "dist")],
                           capture_output=True, text=True, timeout=300)
    assert build.returncode == 0, build.stderr

    wheel, = (tmp_path / "dist").glob("prism_audit-*.whl")
    assert wheel.name.startswith(f"prism_audit-{prism.__version__}-")
    names = set(zipfile.ZipFile(wheel).namelist())
    expected = {f"prism/{relative}" for relative, _ in _prism_sources()}
    assert expected <= names  # every module, including gui/frames, which has no __init__.py
    assert not [n for n in names if n.startswith(("examples/", "tests/"))]
    entry_points = zipfile.ZipFile(wheel).read(f"prism_audit-{prism.__version__}.dist-info/entry_points.txt").decode()
    assert "prism = prism.cli:main" in entry_points
    dist_info = f"prism_audit-{prism.__version__}.dist-info"
    metadata = zipfile.ZipFile(wheel).read(f"{dist_info}/METADATA").decode()
    assert "License-Expression: GPL-3.0-or-later" in metadata
    assert f"{dist_info}/licenses/LICENSE" in names
