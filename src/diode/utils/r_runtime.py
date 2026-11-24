"""Resolve R executables and environments for benchmark subprocesses."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def resolve_rscript(rscript_bin=None):
    if rscript_bin is None:
        candidates = [shutil.which("Rscript"), "/usr/local/bin/Rscript", "/usr/bin/Rscript"]
        rscript_bin = next((path for path in candidates if path and os.path.exists(path)), None)
    elif not os.path.isabs(rscript_bin):
        rscript_bin = shutil.which(rscript_bin) or rscript_bin
    if rscript_bin is None or not os.path.exists(rscript_bin):
        raise FileNotFoundError("Could not find Rscript. Set DIODE_RSCRIPT or configure rscript_bin.")
    return rscript_bin


def r_subprocess_env():
    """Return an environment for R subprocesses without MATLAB runtime paths."""
    env = os.environ.copy()
    ld_library_path = env.get("LD_LIBRARY_PATH")
    if ld_library_path:
        parts = [
            part for part in ld_library_path.split(os.pathsep)
            if "matlab" not in part.lower()
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def resolve_repo_r_runtime(repo_root, package_name, rscript_bin=None):
    """Resolve an R executable and a repository-local package library."""
    root = Path(repo_root)
    library = root / ".r-lib" / package_name
    if rscript_bin in (None, "Rscript"):
        rscript_bin = os.environ.get("DIODE_RSCRIPT") or rscript_bin
    if rscript_bin in (None, "Rscript"):
        saved_paths = (library / "rscript-path", root / ".r-lib" / "alra" / "rscript-path")
        for saved_path in saved_paths:
            if saved_path.is_file():
                saved_rscript = saved_path.read_text().strip()
                if Path(saved_rscript).is_file() and os.access(saved_rscript, os.X_OK):
                    rscript_bin = saved_rscript
                    break
    env = r_subprocess_env()
    if library.is_dir():
        env["R_LIBS"] = os.pathsep.join(filter(None, (str(library), env.get("R_LIBS"))))
    return resolve_rscript(None if rscript_bin == "Rscript" else rscript_bin), env
