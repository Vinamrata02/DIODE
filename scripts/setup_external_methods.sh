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

  if ! command -v Rscript >/dev/null 2>&1; then
    warn "Could not install R dependencies because Rscript is not on PATH. Please follow the R dependency instructions in README.md before running R-backed methods."
    return 0
  fi

  printf 'Checking R dependencies for SCODE/Lasso/MI/IDA...\n'
  if ! Rscript -e "
required <- c('MASS', 'infotheo', 'msgps', 'BiocManager')
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
    warn "Could not install all R dependencies with Rscript at $(command -v Rscript). Please follow the R dependency instructions in README.md before running SCODE, Lasso, MI, or IDA."
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

mkdir -p "${external_dir}"

install_r_dependencies

ensure_external_repo "SCODE" "${scode_dir}" "${scode_repo_url}" "${scode_ref}" "${scode_dir}/SCODE.R"
ensure_external_repo "GRISLI" "${grisli_dir}" "${grisli_repo_url}" "${grisli_ref}" "${grisli_dir}/GRISILI.m"
ensure_external_repo "PHOENIX" "${phoenix_dir}" "${phoenix_repo_url}" "${phoenix_ref}" "${phoenix_dir}/ode_net/code/odenet.py"

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
