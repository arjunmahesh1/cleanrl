#!/usr/bin/env bash
# Example:
#   ALPHA_VALUES_STR="vanilla 210 225 240" SEED_VALUES_STR="1 2 3 4 5" \
#   AXIS_VALUES_STR="friction mass damping" \
#   FACTOR_VALUES_STR="0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.3 1.5 1.7 2.0" \
#   sbatch --array=0-659 slurm/eval_ppo_alpha_grid.sh
#
# The array index maps to (alpha, axis, factor, seed). Each task writes one
# metrics CSV row into a unique file under sweeps/results.

#SBATCH -p compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -J ppo_alpha_eval

set -euo pipefail
shopt -s nullglob

ROOT="${ROOT:-$HOME/cleanrl}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-$HOME/rl_runs_alpha_grid}"
ENV_ID="${ENV_ID:-Walker2d-v4}"
PROJECT="${PROJECT:-fixed-alpha-randomness}"
GROUP="${GROUP:-ppo-alpha-grid-eval}"
ENTITY="${ENTITY:-}"
EXP_PREFIX="${EXP_PREFIX:-ppo_alpha}"
ALGO="${ALGO:-ppo_cont}"
VARIANT_PREFIX="${VARIANT_PREFIX:-a}"
ALPHA_VALUES_STR="${ALPHA_VALUES_STR:-vanilla 210 225 240}"
SEED_VALUES_STR="${SEED_VALUES_STR:-1 2 3 4 5}"
AXIS_VALUES_STR="${AXIS_VALUES_STR:-friction mass damping}"
FACTOR_VALUES_STR="${FACTOR_VALUES_STR:-0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.3 1.5 1.7 2.0}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1000}"
DEVICE="${DEVICE:-cpu}"
TRACK="${TRACK:-false}"
EVAL_RAW_REWARDS="${EVAL_RAW_REWARDS:-true}"
CAPTURE_VIDEO="${CAPTURE_VIDEO:-false}"
XML_OUT_DIR="${XML_OUT_DIR:-$ROOT/perturbed_xml}"
GRAVITY_COMPONENT_INDEX="${GRAVITY_COMPONENT_INDEX:-0}"
GRAVITY_VALUE_PREFIX="${GRAVITY_VALUE_PREFIX:--}"
OUT_STAMP="${OUT_STAMP:-$(date +%Y%m%d)}"
OUT_DIR="${OUT_DIR:-$ROOT/sweeps/results/PPO_AlphaGridEval_${OUT_STAMP}}"
PY="${PY:-}"

read -r -a ALPHA_VALUES <<< "${ALPHA_VALUES_STR}"
read -r -a SEED_VALUES <<< "${SEED_VALUES_STR}"
read -r -a AXIS_VALUES <<< "${AXIS_VALUES_STR}"
read -r -a FACTOR_VALUES <<< "${FACTOR_VALUES_STR}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required. Submit with sbatch --array=0-N."
    exit 1
fi

num_alphas=${#ALPHA_VALUES[@]}
num_seeds=${#SEED_VALUES[@]}
num_axes=${#AXIS_VALUES[@]}
num_factors=${#FACTOR_VALUES[@]}
total_jobs=$((num_alphas * num_axes * num_factors * num_seeds))

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

resolve_alpha_variant() {
    local token="$1"
    case "${token}" in
        vanilla)
            RESOLVED_LABEL="vanilla"
            RESOLVED_CAP=""
            ;;
        noop|a1e9|q1e9|1e9|1000000000|1000000000.0)
            RESOLVED_LABEL="${VARIANT_PREFIX}1e9"
            RESOLVED_CAP="1000000000"
            ;;
        *)
            RESOLVED_LABEL="${VARIANT_PREFIX}$(sanitize_token "${token}")"
            RESOLVED_CAP="${token}"
            ;;
    esac
}

combo_idx=$((SLURM_ARRAY_TASK_ID / num_seeds))
seed_idx=$((SLURM_ARRAY_TASK_ID % num_seeds))
factor_idx=$((combo_idx % num_factors))
combo_idx=$((combo_idx / num_factors))
axis_idx=$((combo_idx % num_axes))
alpha_idx=$((combo_idx / num_axes))

alpha="${ALPHA_VALUES[$alpha_idx]}"
seed="${SEED_VALUES[$seed_idx]}"
axis="${AXIS_VALUES[$axis_idx]}"
factor="${FACTOR_VALUES[$factor_idx]}"

resolve_alpha_variant "${alpha}"
variant_label="${RESOLVED_LABEL}"
exp_name="${EXP_PREFIX}_${variant_label}"

run_dirs=("${TRAIN_RUN_DIR}/${ENV_ID}__${exp_name}__${seed}__"*)
if (( ${#run_dirs[@]} == 0 )); then
    echo "No training run found for exp_name=${exp_name}, seed=${seed} in ${TRAIN_RUN_DIR}"
    exit 1
fi
run_dir="$(ls -1dt "${run_dirs[@]}" | head -n 1)"
model_path="${run_dir}/${exp_name}.cleanrl_model"
norm_stats_path="${model_path}.norm_stats.npz"

if [[ ! -f "${model_path}" ]]; then
    echo "Model not found: ${model_path}"
    exit 1
fi

scenario_label="${axis}_$(sanitize_token "${factor}")"
metrics_dir="${OUT_DIR}/raw_metrics"
metrics_out_csv="${metrics_dir}/${variant_label}_seed_${seed}__${scenario_label}.csv"

axis_args=()
case "${axis}" in
    friction)
        if [[ "${ENV_ID}" == "HalfCheetah-v4" ]]; then
            axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0)
        else
            axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}")
        fi
        ;;
    mass)
        if [[ "${ENV_ID}" == "HalfCheetah-v4" ]]; then
            axis_args+=(--xml-perturb --xml-total-mass-scale "${factor}")
        else
            axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}")
        fi
        ;;
    gear)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}")
        ;;
    gravity)
        axis_args+=(
            --xml-perturb
            --xml-gravity-component-index "${GRAVITY_COMPONENT_INDEX}"
            --xml-gravity-component-value "${GRAVITY_VALUE_PREFIX}${factor}"
        )
        ;;
    thigh_left_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "thigh_left")
        ;;
    leg_left_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "leg_left")
        ;;
    foot_left_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "foot_left")
        ;;
    bthigh_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "bthigh")
        ;;
    bshin_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "bshin")
        ;;
    bfoot_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "bfoot")
        ;;
    fthigh_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "fthigh")
        ;;
    fshin_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "fshin")
        ;;
    ffoot_mass)
        axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-body-name-selector "ffoot")
        ;;
    damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}")
        ;;
    thigh_left_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "thigh_left_joint")
        ;;
    leg_left_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "leg_left_joint")
        ;;
    foot_left_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "foot_left_joint")
        ;;
    bthigh_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "bthigh")
        ;;
    bshin_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "bshin")
        ;;
    bfoot_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "bfoot")
        ;;
    fthigh_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "fthigh")
        ;;
    fshin_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "fshin")
        ;;
    ffoot_damping)
        axis_args+=(--xml-perturb --xml-joint-damping-scale "${factor}" --xml-joint-name-selector "ffoot")
        ;;
    actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}")
        ;;
    thigh_left_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "thigh_left_joint")
        ;;
    leg_left_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "leg_left_joint")
        ;;
    foot_left_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "foot_left_joint")
        ;;
    bthigh_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "bthigh")
        ;;
    bshin_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "bshin")
        ;;
    bfoot_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "bfoot")
        ;;
    fthigh_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "fthigh")
        ;;
    fshin_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "fshin")
        ;;
    ffoot_actuator_gain)
        axis_args+=(--xml-perturb --xml-actuator-gain-scale "${factor}" --xml-actuator-joint-selector "ffoot")
        ;;
    actuator_bias)
        axis_args+=(--xml-perturb --xml-actuator-bias-scale "${factor}")
        ;;
    thigh_left_actuator_bias)
        axis_args+=(--xml-perturb --xml-actuator-bias-scale "${factor}" --xml-actuator-joint-selector "thigh_left_joint")
        ;;
    leg_left_actuator_bias)
        axis_args+=(--xml-perturb --xml-actuator-bias-scale "${factor}" --xml-actuator-joint-selector "leg_left_joint")
        ;;
    foot_left_actuator_bias)
        axis_args+=(--xml-perturb --xml-actuator-bias-scale "${factor}" --xml-actuator-joint-selector "foot_left_joint")
        ;;
    foot_left_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-name-selector "foot_left_geom")
        ;;
    bthigh_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0 --xml-geom-name-selector "bthigh")
        ;;
    bshin_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0 --xml-geom-name-selector "bshin")
        ;;
    bfoot_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0 --xml-geom-name-selector "bfoot")
        ;;
    fthigh_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0 --xml-geom-name-selector "fthigh")
        ;;
    fshin_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0 --xml-geom-name-selector "fshin")
        ;;
    ffoot_friction)
        axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-geom-friction-component 0 --xml-geom-name-selector "ffoot")
        ;;
    obs_noise)
        axis_args+=(--obs-noise-std "${factor}")
        ;;
    state_noise)
        axis_args+=(--obs-noise-std "${factor}")
        ;;
    reward_noise)
        axis_args+=(--reward-noise-std "${factor}")
        ;;
    action_noise)
        axis_args+=(--action-noise-std "${factor}")
        ;;
    action_noise_gaussian)
        axis_args+=(--action-noise-std "${factor}")
        ;;
    action_replace)
        axis_args+=(--action-replace-prob "${factor}")
        ;;
    action_noise_bernoulli|bernoulli_action_noise)
        axis_args+=(--action-replace-prob "${factor}")
        ;;
    friction_mass)
        if [[ "${ENV_ID}" == "HalfCheetah-v4" ]]; then
            axis_args+=(
                --xml-perturb
                --xml-geom-friction-scale "${factor}"
                --xml-geom-friction-component 0
                --xml-total-mass-scale "${factor}"
            )
        else
            axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-body-mass-scale "${factor}")
        fi
        ;;
    friction_damping)
        if [[ "${ENV_ID}" == "HalfCheetah-v4" ]]; then
            axis_args+=(
                --xml-perturb
                --xml-geom-friction-scale "${factor}"
                --xml-geom-friction-component 0
                --xml-joint-damping-scale "${factor}"
            )
        else
            axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-joint-damping-scale "${factor}")
        fi
        ;;
    mass_damping)
        if [[ "${ENV_ID}" == "HalfCheetah-v4" ]]; then
            axis_args+=(--xml-perturb --xml-total-mass-scale "${factor}" --xml-joint-damping-scale "${factor}")
        else
            axis_args+=(--xml-perturb --xml-body-mass-scale "${factor}" --xml-joint-damping-scale "${factor}")
        fi
        ;;
    friction_mass_damping)
        if [[ "${ENV_ID}" == "HalfCheetah-v4" ]]; then
            axis_args+=(
                --xml-perturb
                --xml-geom-friction-scale "${factor}"
                --xml-geom-friction-component 0
                --xml-total-mass-scale "${factor}"
                --xml-joint-damping-scale "${factor}"
            )
        else
            axis_args+=(--xml-perturb --xml-geom-friction-scale "${factor}" --xml-body-mass-scale "${factor}" --xml-joint-damping-scale "${factor}")
        fi
        ;;
    *)
        echo "Unsupported axis: ${axis}"
        exit 1
        ;;
esac

export PYTHONHASHSEED="${seed}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

mkdir -p "${metrics_dir}" "${XML_OUT_DIR}"
cd "${ROOT}"

cmd=(
    "${PY}" evaluate_ppo_robust.py
    --algo "${ALGO}"
    --model-path "${model_path}"
    --norm-stats-path "${norm_stats_path}"
    --env-id "${ENV_ID}"
    --seed "${seed}"
    --device "${DEVICE}"
    --gamma 0.99
    --eval-episodes "${EVAL_EPISODES}"
    --max-episode-steps "${MAX_EPISODE_STEPS}"
    --xml-out-dir "${XML_OUT_DIR}"
    --model-label "${variant_label}"
    --scenario-label "${scenario_label}"
    --metrics-out-csv "${metrics_out_csv}"
    "${axis_args[@]}"
)

if [[ "${EVAL_RAW_REWARDS}" == "true" ]]; then
    cmd+=(--eval-raw-rewards)
else
    cmd+=(--no-eval-raw-rewards)
fi

if [[ "${CAPTURE_VIDEO}" == "true" ]]; then
    cmd+=(--capture-video)
fi

if [[ "${TRACK}" == "true" ]]; then
    cmd+=(--track --wandb-project-name "${PROJECT}" --wandb-group "${GROUP}")
    if [[ -n "${ENTITY}" ]]; then
        cmd+=(--wandb-entity "${ENTITY}")
    fi
fi

echo "[$(date)] eval alpha-grid job"
echo "  env_id=${ENV_ID}"
echo "  exp_name=${exp_name}"
echo "  alpha=${alpha}"
echo "  seed=${seed}"
echo "  axis=${axis}"
echo "  factor=${factor}"
echo "  model_path=${model_path}"
echo "  metrics_out_csv=${metrics_out_csv}"
printf '  command='
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
