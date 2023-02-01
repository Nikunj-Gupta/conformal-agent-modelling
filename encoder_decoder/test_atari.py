from pettingzoo.atari import maze_craze_v3  
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1 

from torch.utils.tensorboard import SummaryWriter 
import torch, numpy as np, argparse, os
import torch.optim as optim 
from torch.utils.tensorboard import SummaryWriter

from itertools import count
from pathlib import Path 

from vae import VAE 
from random_policy import RandomPolicy 

""" 
Hyperparams 
""" 
# multi-agent atari 
frame_size = (64, 64)
has_continuous_action_space = False 
seed = 10 

# VAE 
learning_rate = 0.001
in_channels = 3  
latent_size = 8 
kld_weight = 0.00025 

# RL 
n_agents = 2 
max_episodes = 10_000 
max_timesteps = 100_000 
save_timestep = 5_000 
log_dir = "encoder_decoder/logs/"

# writer = SummaryWriter(log_dir)
log_dir = Path(log_dir)
for i in count(0):
    temp = log_dir/('run{}'.format(i)) 
    if temp.exists():
        pass
    else:
        writer = SummaryWriter(temp)
        log_dir = temp
        break
# set device 
if torch.cuda.is_available(): device = torch.device('cuda')
else: device = torch.device('cpu')

env = maze_craze_v3.parallel_env(
    game_version="race", 
    visibilty_level=0, 
    obs_type='rgb_image', 
    full_action_space=True, 
    max_cycles=max_timesteps,
    auto_rom_install_path="./autorom/roms/" 
)
env = resize_v1(env, frame_size[0], frame_size[1])

# setting seed for reproducibilty 
np.random.seed(seed)
torch.manual_seed(seed)
env.reset(seed=seed) 

state_dim = env.observation_space(env.possible_agents[0]).shape 
action_dim = env.action_space(env.possible_agents[0]).shape[0] if has_continuous_action_space else env.action_space(env.possible_agents[0]).n 

inputs = env.reset() 

# RL agents: Random policy 
agents = [
    RandomPolicy(action_dim, has_continuous_action_space=has_continuous_action_space), 
    RandomPolicy(action_dim, has_continuous_action_space=has_continuous_action_space) 
]

# VAE for learning representations of environment 
vae = VAE(in_channels=in_channels, latent_dim=latent_size, hidden_dims=None).to(device)

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
global_timestep = 0 
for episode in range(1, max_episodes+1): 
    state = env.reset() 
    inputs = agents[0].batchify_obs(state, device).detach().numpy() 
    losses, recon_losses, KL_losses = [], [], [] 
    for timestep in range(1, max_timesteps+1): 
        global_timestep+=1 
        action = {}
        for a in range(n_agents): 
            action[env.possible_agents[a]] = agents[a].select_action(agents[a].batchify_obs(state, device))  
        state, reward, done, is_terminals, info = env.step(action) 
        inputs = np.concatenate((inputs, agents[0].batchify_obs(state, device).detach().numpy())) 
        if timestep % save_timestep == 0: 
            inputs = torch.FloatTensor(inputs).to(device)
            recons = vae.forward(inputs) 
            loss_dict = vae.loss_function(*recons, kld_weight=kld_weight) 
            loss, recon_loss, kl = loss_dict["loss"], loss_dict["Reconstruction_Loss"], loss_dict["KLD"]
            loss.backward()
            optimizer.step()
            l = loss.item() 
            inputs = agents[0].batchify_obs(state, device).detach().numpy() 
        
            writer.add_scalar("1000-Timestep/loss", l, global_timestep) 
            writer.add_scalar("1000-Timestep/reconstruction loss", recon_loss.item(), global_timestep) 
            writer.add_scalar("1000-Timestep/KL", kl.item(), global_timestep) 

            losses.append(l)
            recon_losses.append(recon_loss.item()) 
            KL_losses.append(kl.item()) 
        
            torch.save(vae, log_dir/('vae.pt') )
            print("Episode: {}, Timestep: {}, Loss: {}, Reconstruction loss: {}, KL: {}".format(episode, timestep, l, recon_loss.item(), kl.item())) 
    writer.add_scalar("Episode/loss", np.mean(losses), episode) 
    writer.add_scalar("Episode/reconstruction loss", np.mean(recon_losses), episode) 
    writer.add_scalar("Episode/KL", np.mean(KL_losses), episode) 

