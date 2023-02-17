all: 
	clear 
	# module load python/3.8 
	# pip install -r req.txt 
noam-run: 
	python baselines/baselines.py --envname simple_world_comm_v2 --baseline noam --max_episodes 100000 

giam-run: 
	python baselines/baselines.py --envname simple_world_comm_v2 --baseline giam --max_episodes 100000
