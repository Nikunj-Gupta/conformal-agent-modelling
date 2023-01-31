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

#### Code references 
- [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch) 
- [few-shot-cp](https://github.com/ajfisch/few-shot-cp)
- [LIAM](https://github.com/uoe-agents/LIAM) 
- [VAE](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py)
