all: 
	clear 
	# pip install -r req.txt 
	python ppo/train.py

setup: 
	module load python/3.8 
	source venv/bin/activate 
	# pip install -r req.txt 

noam-run: 
	clear 
	python noam/noam.py 