#!/usr/bin/env bash
# Example:
#   QCAP_VALUES_STR="vanilla 50 75 100" SEED_VALUES_STR="1 2 3 4 5" \
#   sbatch --array=0-19 slurm/train_td3_qcap_grid.sh
#
# The array index maps to (q cap, seed). Use qcap token "vanilla" for
# unclipped TD3; all numeric tokens enable --tv-clip-q-targets.

#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -J td3_qcap_train

set -euo pipefail

ROOT="${ROOT:-$HOME/cleanrl}"
RUN_DIR="${RUN_DIR:-$HOME/rl_runs_td3_qcap_grid}"
ENV_ID="${ENV_ID:-Walker2d-v4}"
PROJECT="${PROJECT:-td3-qcap}"
GROUP="${GROUP:-td3-qcap-grid-train}"
ENTITY="${ENTITY:-}"
EXP_PREFIX="${EXP_PREFIX:-td3_qcap}"
VARIANT_PREFIX="${VARIANT_PREFIX:-q}"
QCAP_VALUES_STR="${QCAP_VALUES_STR:-vanilla 50 75 100}"
SEED_VALUES_STR="${SEED_VALUES_STR:-1 2 3 4 5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
TRACK="${TRACK:-false}"
SAVE_MODEL="${SAVE_MODEL:-true}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
PY="${PY:-}"

read -r -a QCAP_VALUES <<< "${QCAP_VALUES_STR}"
read -r -a SEED_VALUES <<< "${SEED_VALUES_STR}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required. Submit with sbatch --array=0-N."
    exit 1
fi

num_caps=${#QCAP_VALUES[@]}
num_seeds=${#SEED_VALUES[@]}
total_jobs=$((num_caps * num_seeds))

if (( SLURM_ARRAY_TASK_ID >= total_jobs )); then
    echo "Array index ${SLURM_ARRAY_TASK_ID} exceeds grid size ${total_jobs}; exiting."
    exit 1
fi

if [[ -z "${PY}" ]]; then
    if [[ -x "${ROOT}/.venv/bin/python" ]]; then
        PY="${ROOT}/.venv/bin/python"
    else
        PY="python"
    fi
fi

sanitize_token() {
    local token="$1"
    token="${token//./p}"
    token="${token//-/m}"
    echo "${token}"
}

resolve_qcap_variant() {
    local token="$1"
    case "${token}" in
        vanilla)
            RESOLVED_LABEL="vanilla"
            RESOLVED_CAP=""
            ;;
        noop|q1e9|1e9|1000000000|1000000000.0)
            RESOLVED_LABEL="q1e9"
            RESOLVED_CAP="1000000000"
            ;;
        *)
            RESOLVED_LABEL="${VARIANT_PREFIX}$(sanitize_token "${token}")"
            RESOLVED_CAP="${token}"
            ;;
    esac
}

cap_idx=$((SLURM_ARRAY_TASK_ID / num_seeds))
seed_idx=$((SLURM_ARRAY_TASK_ID % num_seeds))
qcap="${QCAP_VALUES[$cap_idx]}"
seed="${SEED_VALUES[$seed_idx]}"

resolve_qcap_variant "${qcap}"
variant_label="${RESOLVED_LABEL}"
resolved_cap="${RESOLVED_CAP}"

exp_name="${EXP_PREFIX}_${variant_label}"

export PYTHONHASHSEED="${seed}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}"
cd "${ROOT}"

cmd=(
    "${PY}" cleanrl/td3_continuous_action.py
    --exp-name "${exp_name}"
    --env-id "${ENV_ID}"
    --seed "${seed}"
    --total-timesteps "${TOTAL_TIMESTEPS}"
    --run-dir "${RUN_DIR}"
)

if [[ "${TRACK}" == "true" ]]; then
    cmd+=(--track --wandb-project-name "${PROJECT}" --wandb-group "${GROUP}")
    if [[ -n "${ENTITY}" ]]; then
        cmd+=(--wandb-entity "${ENTITY}")
    fi
fi

if [[ "${SAVE_MODEL}" == "true" ]]; then
    cmd+=(--save-model)
fi

if [[ "${TORCH_DETERMINISTIC}" == "true" ]]; then
    cmd+=(--torch-deterministic)
else
    cmd+=(--no-torch-deterministic)
fi

if [[ -n "${resolved_cap}" ]]; then
    cmd+=(--tv-clip-q-targets --tv-fixed-cap "${resolved_cap}")
fi

echo "[$(date)] training TD3 qcap-grid job"
echo "  env_id=${ENV_ID}"
echo "  exp_name=${exp_name}"
echo "  qcap=${qcap}"
echo "  seed=${seed}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
echo "  run_dir=${RUN_DIR}"
printf '  command='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
