from pettingzoo.mpe import simple_spread_v2 
from pettingzoo.atari import maze_craze_v3  
from ppo import PPO 
from random_policy import RandomPolicy 
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1 

hyperparams = {
    "max_episodes":2,
    "n_agents": 2, 
    "stack_size":4, 
    "frame_size": (64, 64), 
    "lr_actor": 0.0003, 
    "lr_critic": 0.001, 
    "gamma": 0.99, 
    "K_epochs": 80, 
    "eps_clip": 0.2, 
    "has_continuous_action_space": False, 
    "action_std": 0.6,
    "action_std_init": 0.6 
}


# env = simple_spread_v2.parallel_env(N=hyperparams["n_agents"], continuous_actions=hyperparams["has_continuous_action_space"]) 
env = maze_craze_v3.parallel_env(
    game_version="race", 
    visibilty_level=0, 
    obs_type='rgb_image', 
    full_action_space=True, 
    max_cycles=100000, 
    auto_rom_install_path="./autorom/roms/" 
)
env = color_reduction_v0(env)
env = resize_v1(env, hyperparams["frame_size"][0], hyperparams["frame_size"][1])
env = frame_stack_v1(env, stack_size=hyperparams["stack_size"]) 

state_dim = env.observation_space(env.possible_agents[0]).shape[-1]
state_dim = hyperparams["stack_size"] 
action_dim = env.action_space(env.possible_agents[0]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[0]).n 

state = env.reset() 

agents = [
    PPO(state_dim, action_dim, hyperparams), 
    RandomPolicy(action_dim, hyperparams) 
]

for i in range(1, hyperparams["max_episodes"]+1): 
    action = {}
    for a in range(hyperparams["n_agents"]): 
        action[env.possible_agents[a]] = agents[a].select_action(agents[a].batchify_obs(state, a))  
    state, reward, done, is_terminals, info = env.step(action) 

    print(action) 
