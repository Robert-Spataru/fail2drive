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
python $LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py \
  --routes $F2D_DIR/fail2drive_split/Generalization_PedestriansOnRoad_185.xml \
  --agent /data/robert/fail2drive/team_code/autoagent0_agent.py \
  --agent-config /data/robert/fail2drive/configs/rule_based.yaml