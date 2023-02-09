# conformal-agent-modelling

## Closely related works 
1. [Few-shot Conformal Prediction with Auxiliary Tasks](https://arxiv.org/abs/2102.08898) 
2. [Agent Modelling under Partial Observability for Deep Reinforcement Learning](https://arxiv.org/abs/2006.09447) 


## Appendix   
### Steps for Multiagent ALE Atari 
#### On Local 
Step 1: Install the following: 
```
multi-agent-ale-py
autorom[accept-rom-license]
atari-py
```
Note: Also install `cmake` if not already installed. 

Step 2: Download roms 
```
AutoRom --install-dir <path>
```

Step 3: Import Roms 
```
python -m atari_py.import_roms <path-to-roms> 
```

Good to go! 

#### On Compute canada 
Use AutoROM wheel available in compute canada 
```
avail_wheels autorom

pip install AutoROM==0.1.19 

AutoROM --install-dir <path/to/install/roms>

AutoROM --accept-license 

```

More information: 

Note: If `AutoROM --install-dir <path/to/install/roms>` does not work on compute canada, use trick --> copy roms from local using Globus. 

#### Code references 
- [PPO-Pytorch](https://github.com/nikhilbarhate99/PPO-PyTorch) 
- [few-shot-cp](https://github.com/ajfisch/few-shot-cp)
- [LIAM](https://github.com/uoe-agents/LIAM) 
- [VAE](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py)
- [Conformal-Classification](https://github.com/aangelopoulos/conformal_classification) 
