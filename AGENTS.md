Running the fail2drive benchmark on autoagent0 and other models

1. cd data/robert/fail2drive
2. conda activate ./env
3. source env_vars.sh
4. 
-In terminal 1, type: 
bash /data2/autoagent0/f2d_carla/CarlaUE4.sh -RenderOffScreen
-In terminal 2, type: 
python $LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py \
  --routes $F2D_DIR/fail2drive_split/Generalization_PedestriansOnRoad_185.xml \
  --agent [YOUR_AGENT_FILE] \
  --agent-config [YOUR_AGENT_CONFIG]
E.x:
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --routes ${WORK_DIR}/fail2drive_split/Generalization_PedestriansOnRoad_1085.xml \
  --agent ${WORK_DIR}/team_code/agent_code.py \
  --agent-config ${WORK_DIR}/checkpoints/tfpp

