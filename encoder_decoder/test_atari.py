from pettingzoo.mpe import simple_spread_v2 
from pettingzoo.atari import maze_craze_v3  
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1 
from torch.utils.tensorboard import SummaryWriter 
import torch, numpy as np, argparse, os 


frame_size = (64, 64)
stack_size = 4 
has_continuous_action_space = False 
seed = 10 


env = maze_craze_v3.parallel_env(
    game_version="race", 
    visibilty_level=0, 
    obs_type='rgb_image', 
    full_action_space=True, 
    max_cycles=100_000, 
    auto_rom_install_path="./autorom/roms/" 
)


env = color_reduction_v0(env)
env = resize_v1(env, frame_size[0], frame_size[1])
env = frame_stack_v1(env, stack_size=stack_size) 

# seed
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed) 

state_dim = env.observation_space(env.possible_agents[0]).shape 
# state_dim = stack_size 
action_dim = env.action_space(env.possible_agents[0]).shape[0] if has_continuous_action_space else env.action_space(env.possible_agents[0]).n 
print(state_dim) 

print(action_dim) 

