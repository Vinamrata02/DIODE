#!/usr/bin/env python
"""Preflight checks for optional external benchmark methods."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def command_exists(command):
    return shutil.which(command)


def python_module_exists(module_name):
    try:
        return find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def r_subprocess_env():
    env = os.environ.copy()
    ld_library_path = env.get("LD_LIBRARY_PATH")
    if ld_library_path:
        parts = [
            part for part in ld_library_path.split(os.pathsep)
            if "matlab" not in part.lower()
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def r_package_exists(package_name):
    rscript = command_exists("Rscript")
    if not rscript:
        return False, "Rscript is not on PATH"
    r_code = (
        "user_lib <- Sys.getenv('R_LIBS_USER'); "
        "if (nzchar(user_lib)) .libPaths(c(user_lib, .libPaths())); "
        f"library({package_name}); "
        f"cat('{package_name} ok\\n')"
    )
    result = subprocess.run(
        [rscript, "-e", r_code],
        capture_output=True,
        text=True,
        env=r_subprocess_env(),
    )
    if result.returncode == 0:
        return True, f"R package {package_name} is available"
    lines = (result.stderr or result.stdout or "").strip().splitlines()
    useful_lines = [
        line.strip()
        for line in lines
        if line.strip() and line.strip() != "Execution halted"
    ]
    detail = useful_lines[-1] if useful_lines else f"could not load R package {package_name}"
    return False, detail


def check_paths(*paths):
    missing = [str(path) for path in paths if not Path(path).exists()]
    if missing:
        return False, "missing: " + ", ".join(missing)
    return True, "required files found"


def check_python_packages(*modules):
    missing = [module for module in modules if not python_module_exists(module)]
    if missing:
        return False, "missing Python packages: " + ", ".join(missing)
    return True, "required Python packages import"


def check_r_packages(*packages):
    failures = []
    for package in packages:
        ok, message = r_package_exists(package)
        if not ok:
            failures.append(f"{package} ({message})")
    if failures:
        return False, "missing R packages: " + ", ".join(failures)
    return True, "required R packages load"


def check_matlab_engine():
    matlab_bin = command_exists("matlab")
    if not matlab_bin:
        return False, "matlab is not on PATH"
    if not python_module_exists("matlab.engine"):
        return False, (
            f"matlab found at {matlab_bin}, but MATLAB Engine for Python is not importable; "
            "install MATLAB Engine in this Python environment, for example with "
            "`pip install -e '.[matlab]'` when compatible, or follow README.md"
        )
    return True, f"matlab found at {matlab_bin}; MATLAB Engine imports"


def matlab_mex_extension():
    if sys.platform.startswith("linux"):
        return "mexa64"
    if sys.platform == "darwin":
        return "mexmaci64"
    if sys.platform.startswith("win"):
        return "mexw64"
    return "mex"


def combine_checks(checks):
    failures = [message for ok, message in checks if not ok]
    if failures:
        return "missing", "; ".join(failures)
    details = [message for _, message in checks]
    return "ok", "; ".join(details)


def method_rows():
    external = REPO_ROOT / "external"
    rows = []

    rscript = command_exists("Rscript")
    rscript_check = (bool(rscript), f"Rscript found at {rscript}" if rscript else "Rscript is not on PATH")

    rows.append(("DIODE", "ok", "built-in DIODE method; no external method dependency"))

    phoenix_code = external / "PHOENIX" / "ode_net" / "code"
    rows.append((
        "phoenix",
        *combine_checks([
            check_paths(*(phoenix_code / name for name in (
                "odenet.py", "datahandler.py", "csvreader.py", "figure_saver.py",
                "read_config.py", "torchdiffeq/__init__.py", "torchdiffeq/_impl/adjoint.py",
            ))),
            check_python_packages("numpy", "torch", "matplotlib"),
        ]),
    ))

    rows.append((
        "scode",
        *combine_checks([
            check_paths(external / "SCODE" / "SCODE.R"),
            rscript_check,
            check_r_packages("MASS"),
        ]),
    ))

    rows.append((
        "genie3",
        *combine_checks([
            check_paths(external / "GENIE3" / "GENIE3_python" / "GENIE3.py"),
            check_python_packages("numpy", "pandas", "sklearn"),
        ]),
    ))

    rows.append((
        "grnboost2",
        *combine_checks([
            check_paths(external / "GRNBoost2" / "run_grnboost2.py"),
            check_python_packages("arboreto.core", "dask.dataframe", "distributed"),
        ]),
    ))

    rows.append((
        "dyn_genie3",
        *combine_checks([
            check_paths(external / "dynGENIE3" / "dynGENIE3_python" / "dynGENIE3.py"),
            check_python_packages("numpy", "pandas", "sklearn"),
        ]),
    ))

    grisli_root = external / "GRISLI"
    mex_lasso = grisli_root / "spams-matlab-v2.6" / "build" / f"mexLasso.{matlab_mex_extension()}"
    rows.append((
        "grisli",
        *combine_checks([
            check_paths(grisli_root / "GRISILI.m", grisli_root / "Compute_A_app_wo_ref.m", mex_lasso),
            check_matlab_engine(),
        ]),
    ))

    rows.append((
        "prob",
        *combine_checks([
            check_paths(external / "PROB" / "Executable functions" / "ODE_BayesianLasso_Python.m"),
            check_matlab_engine(),
        ]),
    ))

    rows.append((
        "swing",
        *combine_checks([
            check_paths(external / "SWING" / "Swing" / "Swing.py"),
            check_python_packages("networkx", "numpy", "pandas", "scipy", "sklearn"),
        ]),
    ))

    rows.append((
        "lm",
        *combine_checks([
            check_paths(external / "LR" / "lm.R"),
            rscript_check,
        ]),
    ))

    rows.append((
        "lasso",
        *combine_checks([
            check_paths(external / "Lasso" / "lasso.R"),
            rscript_check,
            check_r_packages("msgps"),
        ]),
    ))

    mi_ida_checks = [
        rscript_check,
        check_r_packages("infotheo", "pcalg"),
    ]
    rows.append((
        "mi",
        *combine_checks([
            check_paths(external / "MI" / "run_mi_ida.R"),
            *mi_ida_checks,
        ]),
    ))
    rows.append((
        "ida_min",
        *combine_checks([
            check_paths(external / "IDA" / "run_mi_ida.R"),
            *mi_ida_checks,
        ]),
    ))
    rows.append((
        "ida_mean",
        *combine_checks([
            check_paths(external / "IDA" / "run_mi_ida.R"),
            *mi_ida_checks,
        ]),
    ))

    return rows


def print_table(rows):
    headers = ("method", "status", "message")
    widths = [
        max(len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(len(headers))
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def main():
    rows = method_rows()
    print_table(rows)
    return 1 if any(status != "ok" for _, status, _ in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
