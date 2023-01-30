from pettingzoo.mpe import simple_spread_v2 
from ppo_mpe import PPO 
from random_policy import RandomPolicy 
from torch.utils.tensorboard import SummaryWriter 
import torch, numpy as np, argparse, os 


parser = argparse.ArgumentParser() 
parser.add_argument("--seed", type=int) 
parser.add_argument("--n_agents", type=int)
parser.add_argument("--log_dir", type=str)
args = parser.parse_known_args()[0] 

log_name = "noam"  
log_name.append("n_" + str(args.n_agents)) 
log_name.append("seed_" + str(args.seed)) 
log_name = "--".join(log_name) 

hyperparams = {
    "n_agents": args.n_agents, 
    "max_episodes":30_000,
    "max_cycles":25,
    "update_timestep": 30, 
    "save_model_freq": 5_000, 
    "logs_dir": os.path.join(args.log_dir, log_name), 
    "action_std_decay_rate": 0.05, 
    "action_std_decay_freq": int(2.5e5), 
    "min_action_std": 0.1, 
    "lr_actor": 0.0003, 
    "lr_critic": 0.001, 
    "gamma": 0.95, 
    "K_epochs": 8, 
    "eps_clip": 0.2, 
    "has_continuous_action_space": False, 
    "action_std": 0.5,
    "action_std_init": 0.6 
}
env = simple_spread_v2.parallel_env(
    N=hyperparams["n_agents"], 
    max_cycles=hyperparams["max_cycles"],
    continuous_actions=hyperparams["has_continuous_action_space"]
)
# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed) 

state_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_dim = env.action_space(env.possible_agents[0]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[0]).n 

agents = [
    PPO(state_dim, action_dim, hyperparams), 
    # PPO(state_dim, action_dim, hyperparams), 
    RandomPolicy(action_dim, hyperparams) 
]

writer = SummaryWriter(hyperparams["logs_dir"])

for i_episode in range(1, hyperparams["max_episodes"]+1): 

    state = env.reset() 
    ep_reward = 0 

    for t in range(1, hyperparams["max_cycles"]+1):
        action = {}
        for a in range(hyperparams["n_agents"]): 
            action[env.possible_agents[a]] = agents[a].select_action(state[env.possible_agents[a]]) 
        state, reward, done, is_terminals, info = env.step(action) 

        # saving reward and is_terminals
        for a in range(hyperparams["n_agents"]): 
            if isinstance(agents[a], PPO): 
                agents[a].buffer.rewards.append(reward[env.possible_agents[a]])
                agents[a].buffer.is_terminals.append(done[env.possible_agents[a]])
        ep_reward += sum([reward[env.possible_agents[a]] for a in range(hyperparams["n_agents"])])/hyperparams["n_agents"]  # mean 

        # if continuous action space; then decay action std of ouput action distribution
        if hyperparams["has_continuous_action_space"] and t % hyperparams["action_std_decay_freq"] == 0:
            for a in range(hyperparams["n_agents"]): 
                if isinstance(agents[a], PPO): 
                    agents[a].decay_action_std(hyperparams["action_std_decay_rate"], hyperparams["min_action_std"]) 
        
        if any(list(done.values())): 
            break 
            
    # update PPO agent
    if i_episode % hyperparams["update_timestep"] == 0:
        for a in range(hyperparams["n_agents"]): 
            if isinstance(agents[a], PPO): 
                agents[a].update() 

    print("Episode : {} \t\t Average Reward : {}".format(i_episode, ep_reward)) 
    writer.add_scalar("Episodic Return", ep_reward, i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        for a in range(hyperparams["n_agents"]): 
            if isinstance(agents[a], PPO): 
                print("Saving model at episode: ", i_episode) 
                checkpoint_path = hyperparams["logs_dir"] + "/agent_" + str(a) + '.pth'
                agents[a].save(checkpoint_path) 

