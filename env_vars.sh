LOCAL="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
CARLA="/data2/autoagent0/f2d_carla"
ENV_PATH="$LOCAL/env"

# Use -p (path) instead of -n (name) to target your local environment folder
conda env config vars set WORK_DIR=$LOCAL -p $ENV_PATH
conda env config vars set CARLA_ROOT=$CARLA -p $ENV_PATH

conda env config vars set LEADERBOARD_ROOT=$LOCAL/leaderboard -p $ENV_PATH
conda env config vars set SCENARIO_RUNNER_ROOT=$LOCAL/scenario_runner -p $ENV_PATH

conda env config vars set PYTHONPATH=$CARLA/PythonAPI/carla:$LOCAL/leaderboard:$LOCAL/scenario_runner -p $ENV_PATH

# Deactivate and reactivate your local environment path to apply the changes
conda deactivate
conda activate $ENV_PATH