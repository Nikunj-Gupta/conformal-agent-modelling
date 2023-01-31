all: 
	clear 
	# pip install -r req.txt 
	# python ppo_mpe_hammer/train.py --seed 3 --n_agents 3 --meslen 0 
	# python ppo_mpe_hammer/gen_runs.py 
	# python ppo_mpe_hammer/save_np.py 
	# python ppo_mpe_hammer/plot.py 
	python encoder_decoder/test.py 

setup: 
	module load python/3.8 
	source venv/bin/activate 
	# pip install -r req.txt 

noam-run: 
	clear 
	python noam/noam.py 

run-1: 
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 2 --meslen 0  
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 2 --meslen 1 
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 5 --meslen 0  
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 5 --meslen 1 

run-2: 
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 3 --meslen 0  
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 3 --meslen 1 
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 7 --meslen 0  
	python ppo_mpe_hammer/train.py --seed 10 --n_agents 7 --meslen 1   