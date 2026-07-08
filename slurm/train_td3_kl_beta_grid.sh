#!/usr/bin/env bash
# Example:
#   BETA_VALUES_STR="vanilla 0.5 1 2 5 10 20 50 100" SEED_VALUES_STR="1 2 3 4 5" \
#   sbatch --array=0-39 slurm/train_td3_kl_beta_grid.sh
#
# The array index maps to (KL beta, seed). Use beta token "vanilla" for
# ordinary TD3; all numeric tokens enable --robust-target-mode kl_moment.

#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH -J td3_kl_train

set -euo pipefail

ROOT="${ROOT:-$HOME/cleanrl}"
RUN_DIR="${RUN_DIR:-$HOME/rl_runs_td3_kl_beta_grid}"
ENV_ID="${ENV_ID:-Walker2d-v4}"
PROJECT="${PROJECT:-td3-kl-moment}"
GROUP="${GROUP:-td3-kl-beta-grid-train}"
ENTITY="${ENTITY:-}"
EXP_PREFIX="${EXP_PREFIX:-td3_kl}"
VARIANT_PREFIX="${VARIANT_PREFIX:-klb}"
BETA_VALUES_STR="${BETA_VALUES_STR:-vanilla 0.5 1 2 5 10 20 50 100}"
SEED_VALUES_STR="${SEED_VALUES_STR:-1 2 3 4 5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-1000000}"
REWARD_SCALE="${REWARD_SCALE:-0.01}"
KL_LOG_MOMENT_EXP_MIN="${KL_LOG_MOMENT_EXP_MIN:--80}"
KL_LOG_MOMENT_EXP_MAX="${KL_LOG_MOMENT_EXP_MAX:-20}"
TRACK="${TRACK:-false}"
SAVE_MODEL="${SAVE_MODEL:-true}"
TORCH_DETERMINISTIC="${TORCH_DETERMINISTIC:-true}"
PY="${PY:-}"

read -r -a BETA_VALUES <<< "${BETA_VALUES_STR}"
read -r -a SEED_VALUES <<< "${SEED_VALUES_STR}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required. Submit with sbatch --array=0-N."
    exit 1
fi

num_betas=${#BETA_VALUES[@]}
num_seeds=${#SEED_VALUES[@]}
total_jobs=$((num_betas * num_seeds))

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

resolve_beta_variant() {
    local token="$1"
    case "${token}" in
        vanilla)
            RESOLVED_LABEL="vanilla"
            RESOLVED_BETA=""
            ;;
        *)
            RESOLVED_LABEL="${VARIANT_PREFIX}$(sanitize_token "${token}")"
            RESOLVED_BETA="${token}"
            ;;
    esac
}

beta_idx=$((SLURM_ARRAY_TASK_ID / num_seeds))
seed_idx=$((SLURM_ARRAY_TASK_ID % num_seeds))
beta="${BETA_VALUES[$beta_idx]}"
seed="${SEED_VALUES[$seed_idx]}"

resolve_beta_variant "${beta}"
variant_label="${RESOLVED_LABEL}"
resolved_beta="${RESOLVED_BETA}"

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
    --reward-scale "${REWARD_SCALE}"
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

if [[ -n "${resolved_beta}" ]]; then
    cmd+=(
        --robust-target-mode kl_moment
        --kl-beta "${resolved_beta}"
        --kl-log-moment-exp-min "${KL_LOG_MOMENT_EXP_MIN}"
        --kl-log-moment-exp-max "${KL_LOG_MOMENT_EXP_MAX}"
    )
fi

echo "[$(date)] training TD3 KL beta-grid job"
echo "  env_id=${ENV_ID}"
echo "  exp_name=${exp_name}"
echo "  beta=${beta}"
echo "  seed=${seed}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
echo "  reward_scale=${REWARD_SCALE}"
echo "  run_dir=${RUN_DIR}"
printf '  command='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
