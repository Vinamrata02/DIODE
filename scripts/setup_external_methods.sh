#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_dir="${repo_root}/external"

scode_repo_url="${SCODE_REPO_URL:-https://github.com/hmatsu1226/SCODE.git}"
scode_ref="${SCODE_REF:-d481cd807b9b44e8148d540cdac15cbcae159478}"
scode_dir="${external_dir}/SCODE"
grisli_repo_url="${GRISLI_REPO_URL:-https://github.com/PCAubin/GRISLI.git}"
grisli_ref="${GRISLI_REF:-fc7ac01f2b09e0effceb900ba65c8aa80a0ce9a9}"
grisli_dir="${external_dir}/GRISLI"
grisli_compile_spams="${GRISLI_COMPILE_SPAMS:-auto}"
phoenix_repo_url="${PHOENIX_REPO_URL:-https://github.com/QuackenbushLab/phoenix.git}"
phoenix_ref="${PHOENIX_REF:-b18317663aada933fa26402df6aad86d042bd13b}"
phoenix_dir="${external_dir}/PHOENIX"
install_r_deps="${DIODE_INSTALL_R_DEPS:-1}"
refresh_external="${DIODE_REFRESH_EXTERNAL:-0}"

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

install_r_dependencies() {
  if [[ "${install_r_deps}" == "0" ]]; then
    printf 'Skipping R dependency installation because DIODE_INSTALL_R_DEPS=0.\n'
    return 0
  fi

  local rscript
  local library="${repo_root}/.r-lib/scode"
  if ! rscript="$(resolve_setup_rscript "${library}")"; then
    warn "Rscript is unavailable. Set DIODE_RSCRIPT or put Rscript on PATH before installing R dependencies."
    return 0
  fi

  mkdir -p "${library}"
  if PATH="$(dirname "${rscript}"):${PATH}" R_LIBS="${library}${R_LIBS:+:${R_LIBS}}" \
    "${rscript}" --vanilla - "${library}" <<'RS'
lib <- commandArgs(trailingOnly = TRUE)[1L]
if (!requireNamespace("MASS", quietly = TRUE)) {
  install.packages("MASS", lib = lib, repos = "https://cloud.r-project.org")
}
stopifnot(requireNamespace("MASS", quietly = TRUE))
RS
  then
    "${rscript}" --vanilla -e 'cat(normalizePath(commandArgs(trailingOnly = TRUE)[1]), "\n", sep="")' \
      "${rscript}" > "${library}/rscript-path"
  else
    warn "Could not install MASS for SCODE with Rscript at ${rscript}."
  fi

  printf 'Checking R dependencies for Lasso/MI/IDA...\n'
  if ! "${rscript}" -e "
required <- c('infotheo', 'msgps', 'BiocManager')
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  install.packages(missing, repos='https://cloud.r-project.org')
}
if (!requireNamespace('graph', quietly = TRUE) || !requireNamespace('RBGL', quietly = TRUE)) {
  BiocManager::install(c('graph', 'RBGL'), ask=FALSE, update=FALSE)
}
if (!requireNamespace('pcalg', quietly = TRUE)) {
  install.packages('pcalg', repos='https://cloud.r-project.org')
}
" ; then
    warn "Could not install all R dependencies with Rscript at ${rscript}. Please follow the R dependency instructions in README.md before running Lasso, MI, or IDA."
  fi
}

ensure_external_repo() {
  local label="$1"
  local dir="$2"
  local url="$3"
  local ref="$4"
  local required_file="$5"

  if [[ -f "${required_file}" && "${refresh_external}" != "1" ]]; then
    printf '%s is already available at %s\n' "${label}" "${dir}"
    return 0
  fi

  if [[ -d "${dir}/.git" ]]; then
    printf 'Refreshing %s at %s because DIODE_REFRESH_EXTERNAL=1 or required files are missing.\n' "${label}" "${dir}"
    git -C "${dir}" fetch --all --tags
    git -C "${dir}" checkout "${ref}"
    return 0
  fi

  if [[ -d "${dir}" ]]; then
    warn "${label} directory exists at ${dir}, but required file ${required_file} is missing and it is not a git checkout. Please restore it manually or move the directory before rerunning setup."
    return 0
  fi

  git clone "${url}" "${dir}"
  git -C "${dir}" checkout "${ref}"
}

resolve_setup_rscript() {
  local requested_rscript="${DIODE_RSCRIPT:-}"
  local file saved_rscript
  if [[ -z "${requested_rscript}" ]]; then
    for file in "$1/rscript-path" "${repo_root}/.r-lib/alra/rscript-path"; do
      if [[ -s "${file}" ]]; then
        saved_rscript="$(<"${file}")"
        if [[ -f "${saved_rscript}" && -x "${saved_rscript}" ]]; then
          requested_rscript="${saved_rscript}"
          break
        fi
      fi
    done
  fi
  command -v "${requested_rscript:-Rscript}"
}

install_alra() {
  if [[ "${install_r_deps}" == "0" ]]; then
    return 0
  fi

  local rscript
  local library="${repo_root}/.r-lib/alra"
  local url="https://github.com/KlugerLab/ALRA/archive/f34d46570b1221179047d9f99c235ed880cc3bae.tar.gz"
  if ! rscript="$(resolve_setup_rscript "${library}")"; then
    warn "ALRA requires an existing Rscript. Set DIODE_RSCRIPT to its executable path."
    return 0
  fi

  mkdir -p "${library}"
  if ! R_LIBS="${library}${R_LIBS:+:${R_LIBS}}" "${rscript}" --vanilla - "${library}" "${url}" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
lib <- args[1L]
if (!requireNamespace("rsvd", quietly = TRUE)) {
  install.packages("rsvd", lib = lib, repos = "https://cloud.r-project.org")
}
if (!requireNamespace("ALRA", quietly = TRUE)) {
  install.packages(args[2L], lib = lib, repos = NULL, type = "source")
}
stopifnot(requireNamespace("ALRA", quietly = TRUE))
RS
  then
    warn "Could not install ALRA with Rscript at ${rscript}."
    return 0
  fi

  "${rscript}" --vanilla -e 'cat(normalizePath(commandArgs(trailingOnly = TRUE)[1]), "\n", sep="")' \
    "${rscript}" > "${library}/rscript-path"
}

install_slingshot() {
  if [[ "${install_r_deps}" == "0" ]]; then
    return 0
  fi

  local rscript
  local library="${repo_root}/.r-lib/slingshot"
  if ! rscript="$(resolve_setup_rscript "${library}")"; then
    warn "Fig. 7 requires Rscript for Slingshot. Set DIODE_RSCRIPT to its executable path."
    return 0
  fi

  mkdir -p "${library}"
  if ! PATH="$(dirname "${rscript}"):${PATH}" R_LIBS="${library}${R_LIBS:+:${R_LIBS}}" \
    "${rscript}" - "${library}" <<'RS'
lib <- commandArgs(trailingOnly = TRUE)[1L]
required <- c("slingshot", "DelayedMatrixStats")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", lib = lib, repos = "https://cloud.r-project.org")
  }
  BiocManager::install(missing, lib = lib, ask = FALSE, update = FALSE, dependencies = NA)
}
stopifnot(all(vapply(required, requireNamespace, logical(1), quietly = TRUE)))
RS
  then
    warn "Could not install the Slingshot R package at ${library}."
    return 0
  fi

  "${rscript}" --vanilla -e 'cat(normalizePath(commandArgs(trailingOnly = TRUE)[1]), "\n", sep="")' \
    "${rscript}" > "${library}/rscript-path"
}

install_imputation_methods() {
  local python_bin="${DIODE_PYTHON:-python}"
  local scnode_dir="${DIODE_SCNODE_ROOT:-${external_dir}/scNODE}"
  local mioflow_dir="${DIODE_MIOFLOW_ROOT:-${external_dir}/MIOFlow}"
  ensure_external_repo "scNODE" "${scnode_dir}" https://github.com/rsinghlab/scNODE.git \
    ba55e8ddecfb74874047517952ab799326b4126d "${scnode_dir}/optim/running.py"
  ensure_external_repo "MIOFlow" "${mioflow_dir}" https://github.com/KrishnaswamyLab/MIOFlow.git \
    ec4b8ba06784aeba0f6e2be6c6204a8e806e1c77 "${mioflow_dir}/mioflow/mioflow.py"

  "${python_bin}" -m pip install -r "${repo_root}/requirements.txt"
  if "${python_bin}" -c 'import sys; sys.exit(sys.version_info < (3, 12))'; then
    # scprep's old pandas constraint would replace the existing pandas build.
    "${python_bin}" -m pip install --no-deps magic-impute==3.0.0 scprep==1.2.3
  fi
  "${python_bin}" -c 'import magic, scanpy, geomloss, torchdiffeq, torchsde, ot, phate'
  install_alra
  install_slingshot
}

mkdir -p "${external_dir}"

install_r_dependencies

ensure_external_repo "SCODE" "${scode_dir}" "${scode_repo_url}" "${scode_ref}" "${scode_dir}/SCODE.R"
ensure_external_repo "GRISLI" "${grisli_dir}" "${grisli_repo_url}" "${grisli_ref}" "${grisli_dir}/GRISILI.m"
ensure_external_repo "PHOENIX" "${phoenix_dir}" "${phoenix_repo_url}" "${phoenix_ref}" "${phoenix_dir}/ode_net/code/odenet.py"
install_imputation_methods

# GRNBoost2 is packaged by Arboreto, so no separate git checkout is needed.
grnboost2_python="${GRNBOOST2_PYTHON:-python}"
if [[ "${DIODE_INSTALL_GRNBOOST2:-0}" == "1" ]]; then
  "${grnboost2_python}" -m pip install -r "${external_dir}/GRNBoost2/requirements.txt"
fi
if ! "${grnboost2_python}" -c "import arboreto.core, dask.dataframe, distributed" >/dev/null 2>&1; then
  warn "GRNBoost2 dependencies are missing. Run python -m pip install -e '.[grnboost2]' from the repository root, or rerun setup with DIODE_INSTALL_GRNBOOST2=1. GRNBOOST2_PYTHON selects the interpreter."
fi

# PHOENIX's integrator is bundled upstream; its remaining Python dependencies
# (NumPy, PyTorch, Matplotlib) are installed by pip install -e . / environment.yml.
phoenix_python="${PHOENIX_PYTHON:-python}"
if ! "${phoenix_python}" -c "import numpy, torch, matplotlib" >/dev/null 2>&1; then
  warn "PHOENIX requires NumPy, PyTorch, and Matplotlib in the active Python environment. Install this package with python -m pip install -e .; set PHOENIX_PYTHON to select the interpreter for this check."
fi
for phoenix_file in datahandler.py csvreader.py figure_saver.py read_config.py torchdiffeq/__init__.py torchdiffeq/_impl/adjoint.py; do
  if [[ ! -f "${phoenix_dir}/ode_net/code/${phoenix_file}" ]]; then
    warn "PHOENIX runtime is incomplete: missing ${phoenix_file}. Restore external/PHOENIX from the repository before running PHOENIX."
  fi
done

spams_dir="${grisli_dir}/spams-matlab-v2.6"
mex_ext="mex"
case "$(uname -s)" in
  Linux*) mex_ext="mexa64" ;;
  Darwin*) mex_ext="mexmaci64" ;;
  MINGW*|MSYS*|CYGWIN*) mex_ext="mexw64" ;;
esac
mex_lasso="${spams_dir}/build/mexLasso.${mex_ext}"
if [[ "${grisli_compile_spams}" != "0" && ! -f "${mex_lasso}" ]]; then
  if command -v matlab >/dev/null 2>&1; then
    printf 'Compiling GRISLI SPAMS MATLAB MEX files for %s...\n' "${mex_ext}"
    if ! matlab -batch "cd('${spams_dir}'); compile; start_spams; assert(exist('mexLasso','file') > 0);"; then
      warn "Could not compile GRISLI SPAMS MEX files at ${spams_dir}. Please follow the GRISLI/SPAMS instructions in README.md before running GRISLI."
    fi
  else
    warn "Could not compile GRISLI SPAMS because matlab is not on PATH. Please follow the MATLAB/GRISLI instructions in README.md before running GRISLI."
    printf 'Set GRISLI_COMPILE_SPAMS=0 to skip this check.\n'
  fi
elif [[ -f "${mex_lasso}" ]]; then
  printf 'GRISLI SPAMS MEX is already available at %s\n' "${mex_lasso}"
fi
