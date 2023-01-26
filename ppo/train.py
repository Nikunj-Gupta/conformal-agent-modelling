from pettingzoo.mpe import simple_spread_v2 
from pettingzoo.atari import maze_craze_v3  
from ppo import PPO 
from random_policy import RandomPolicy 
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1 
from torch.utils.tensorboard import SummaryWriter


hyperparams = {
    "max_episodes":100_000,
    "max_cycles":100_000,
    "update_timestep": 4, 
    "save_model_freq": int(100_000/20), 
    "logs_dir": "ppo/logs/ppo_vs_random/", 
    "action_std_decay_rate": 0.05, 
    "action_std_decay_freq": int(2.5e5), 
    "min_action_std": 0.1, 
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

agents = [
    PPO(state_dim, action_dim, hyperparams), 
    RandomPolicy(action_dim, hyperparams) 
]

writer = SummaryWriter(hyperparams["logs_dir"])

for i_episode in range(1, hyperparams["max_episodes"]+1): 

    state = env.reset() 
    ep_reward = 0 

    for t in range(1, hyperparams["max_cycles"]+1):
        action = {}
        for a in range(hyperparams["n_agents"]): 
            action[env.possible_agents[a]] = agents[a].select_action(agents[a].batchify_obs(state, a))  
        state, reward, done, is_terminals, info = env.step(action) 

        # saving reward and is_terminals
        for a in range(hyperparams["n_agents"]): 
            if isinstance(agents[a], PPO): 
                agents[a].buffer.rewards.append(reward[env.possible_agents[a]])

        ep_reward += sum([reward[env.possible_agents[a]] for a in range(hyperparams["n_agents"])]) 

        # if continuous action space; then decay action std of ouput action distribution
        if hyperparams["has_continuous_action_space"] and t % hyperparams["action_std_decay_freq"] == 0:
            for a in range(hyperparams["n_agents"]): 
                if isinstance(agents[a], PPO): 
                    agents[a].decay_action_std(hyperparams["action_std_decay_rate"], hyperparams["min_action_std"]) 
        
        if t % 10_000 == 0: 
            print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, t, ep_reward)) 

        if any(list(done.values())): 
            break 
            
    # update PPO agent
    if i_episode % hyperparams["update_timestep"] == 0:
        for a in range(hyperparams["n_agents"]): 
            if isinstance(agents[a], PPO): 
                agents[a].update() 

    print("Episode : {} \t\t Average Reward : {}".format(i_episode, ep_reward)) 
    writer.add_scalar("Episodic Return", ep_reward, i_episode) 
    writer.add_scalar("Episode Running Timesteps", t, i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        for a in range(hyperparams["n_agents"]): 
            if isinstance(agents[a], PPO): 
                print("Saving model at episode: ", i_episode) 
                checkpoint_path = hyperparams["logs_dir"] + "/agent_" + str(a) + '.pth'
                agents[a].save(checkpoint_path) 
