all: 
	clear 
	# module load python/3.8 
	# pip install -r req.txt 
	# python conformal-action-prediction/conformal-rl-reps.py 
	python baselines/baselines.py --envname simple_tag_v2 --baseline noam --max_episodes 30000 --n_adversaries 3 
noam-run: 
	python baselines/baselines.py --envname simple_adversary_v2 --baseline noam --max_episodes 1000 --save_model_freq 100 
	python baselines/baselines.py --envname simple_tag_v2 --baseline noam --n_adversaries 3 --max_episodes 1000 --save_model_freq 100 
	python baselines/baselines.py --envname simple_world_comm_v2 --baseline noam --n_adversaries 4 --max_episodes 1000 --save_model_freq 100 

giam-run: 
	python baselines/baselines.py --envname simple_adversary_v2 --baseline giam --max_episodes 1000 --save_model_freq 100 
	python baselines/baselines.py --envname simple_tag_v2 --baseline giam --n_adversaries 3 --max_episodes 1000 --save_model_freq 100 
	python baselines/baselines.py --envname simple_world_comm_v2 --baseline giam --n_adversaries 4 --max_episodes 1000 --save_model_freq 100 
