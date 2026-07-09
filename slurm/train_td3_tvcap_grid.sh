#!/usr/bin/env bash
# Example:
#   TVCAP_VALUES_STR="vanilla 250 350 450 550" SEED_VALUES_STR="1 2 3 4 5" \
#   sbatch --array=0-24 slurm/train_td3_tvcap_grid.sh
#
# The array index maps to (TV Q-target cap, seed). Use cap token "vanilla" for
# ordinary TD3; all numeric tokens enable --robust-target-mode tv_cap.

#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -J td3_tvcap_train

set -euo pipefail

ROOT="${ROOT:-$HOME/cleanrl}"
RUN_DIR="${RUN_DIR:-$HOME/rl_runs_td3_tvcap_grid}"
ENV_ID="${ENV_ID:-Walker2d-v4}"
PROJECT="${PROJECT:-td3-experiments}"
GROUP="${GROUP:-td3-tvcap-grid-train}"
ENTITY="${ENTITY:-}"
EXP_PREFIX="${EXP_PREFIX:-td3_tvcap}"
VARIANT_PREFIX="${VARIANT_PREFIX:-tvc}"
TVCAP_VALUES_STR="${TVCAP_VALUES_STR:-vanilla 250 350 450 550}"
SEED_VALUES_STR="${SEED_VALUES_STR:-1 2 3 4 5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
TRACK="${TRACK:-false}"
SAVE_MODEL="${SAVE_MODEL:-true}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
PY="${PY:-}"

read -r -a TVCAP_VALUES <<< "${TVCAP_VALUES_STR}"
read -r -a SEED_VALUES <<< "${SEED_VALUES_STR}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required. Submit with sbatch --array=0-N."
    exit 1
fi

num_caps=${#TVCAP_VALUES[@]}
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

resolve_tvcap_variant() {
    local token="$1"
    case "${token}" in
        vanilla)
            RESOLVED_LABEL="vanilla"
            RESOLVED_CAP=""
            ;;
        noop|tvc1e9|q1e9|1e9|1000000000|1000000000.0)
            RESOLVED_LABEL="${VARIANT_PREFIX}1e9"
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
tvcap="${TVCAP_VALUES[$cap_idx]}"
seed="${SEED_VALUES[$seed_idx]}"

resolve_tvcap_variant "${tvcap}"
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
    cmd+=(
        --robust-target-mode tv_cap
        --tv-fixed-cap "${resolved_cap}"
    )
fi

echo "[$(date)] training TD3 TV-cap grid job"
echo "  env_id=${ENV_ID}"
echo "  exp_name=${exp_name}"
echo "  tvcap=${tvcap}"
echo "  seed=${seed}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
echo "  run_dir=${RUN_DIR}"
printf '  command='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
