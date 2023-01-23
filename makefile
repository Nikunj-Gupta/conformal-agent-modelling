all: 
	clear 
	pip install -r req.txt 

setup: 
	module load python/3.8 
	source venv/bin/activate 

noam: 
	clear 
	python NOAM/noam.py 