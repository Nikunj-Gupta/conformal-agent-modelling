# conformal-agent-modelling
[Few-shot Conformal Prediction with Auxiliary Tasks](https://arxiv.org/abs/2102.08898) 



### Appendix   
#### Steps for Multiagent ALE Atari 
Step 1: Install the following: 
```
multi-agent-ale-py
autorom[accept-rom-license]
atari-py
```
Also install `cmake` if not already installed. 

Step 2: Download roms 
```
AutoRom --install-dir <path>
```

Step 3: Import Roms 
```
python -m atari_py.import_roms <path-to-roms> 
```

Good to go! 