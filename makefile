all: 
	clear 
	# pip install -r req.txt 
	# python encoder_decoder/vae.py 
	# time python encoder_decoder/test_atari.py 
	python giam/giam.py --n_agents 2 --log_dir giam/logs --seed 10 

setup: 
	module load python/3.8 
	source venv/bin/activate 
	# pip install -r req.txt 

noam-run-random: 
	clear 
	for number in 10 22 37 48 ; do \
    	python ppo_mpe/train.py --n_agents 2 --random_other_agent 1 --log_dir final_logs/ --seed $$number ; \
	done

noam-run-marl: 
	clear 
	for number in 10 22 37 48 ; do \
    	python ppo_mpe/train.py --n_agents 2 --random_other_agent 0 --log_dir final_logs/ --seed $$number ; \
	done 

giam-run-marl: 
	clear 
	for number in 10 22 37 48 ; do \
    	python giam/giam.py --n_agents 2 --log_dir final_logs/ --seed $$number ; \
	done
	