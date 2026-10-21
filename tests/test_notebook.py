import re
import subprocess
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path("demo.ipynb")


@pytest.fixture(scope="module")
def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_every_code_cell_was_run_and_none_ended_in_an_error(notebook):
    cells = code_cells(notebook)
    assert cells and all(cell.execution_count for cell in cells)
    errors = [o for cell in cells for o in cell.outputs if o.output_type == "error"]
    assert errors == []


def test_the_saved_outputs_hold_no_local_paths_names_or_warnings(notebook):
    text = " ".join(str(o) for cell in code_cells(notebook) for o in cell.outputs)
    assert not re.search(r"/mnt/|/home/|/Users/|Warning", text)


def test_the_notebook_reads_only_files_that_exist_and_are_not_ignored_by_git(notebook):
    source = "\n".join(cell.source for cell in code_cells(notebook))
    files = set(re.findall(r'RES / "([\w.]+)"', source)) | set(re.findall(r'"((?:results|docs/img)/[\w./]+)"', source))
    paths = [Path("results") / f if "/" not in f else Path(f) for f in files]
    assert len(paths) >= 15
    for path in paths:
        assert path.exists(), path
        ignored = subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode == 0
        assert not ignored, f"{path} is ignored by git, so a fresh clone would not have it"


def test_the_notebook_does_not_read_the_database_the_holdout_data_or_row_level_files(notebook):
    source = "\n".join(cell.source for cell in code_cells(notebook))
    for forbidden in ("sqlite", "home_credit.db", "shap_sample", "shap_applicants.csv", "weighted_shift_sample", "svm_margins", "oof_best", "mlflow"):
        assert forbidden not in source
