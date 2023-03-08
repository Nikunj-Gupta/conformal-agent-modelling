import lbforaging, gym
import torch, numpy as np, argparse, os 

from ppo import PPO 
from torch.utils.tensorboard import SummaryWriter 
from itertools import count
from pathlib import Path 

parser = argparse.ArgumentParser() 

parser.add_argument("--env", type=str, default="Foraging-12x12-2p-4f-coop-v1") 
parser.add_argument("--baseline", type=str) 
parser.add_argument("--max_episodes", type=int, default=500_000) 
parser.add_argument("--seed", type=int, default=0) 
parser.add_argument("--log_dir", type=str, default="./debug_logs/lbf/") 

args = parser.parse_known_args()[0] 

log_name = [
    args.env, 
    args.baseline
] 
log_name.append("seed_" + str(args.seed)) 
log_name = "--".join(log_name) 

# env = gym.make("Foraging-8x8-2p-1f-coop-v1")
env = gym.make(args.env)
env.reset() 

hyperparams = {
    "n_agents": env.n_agents, 
    "max_episodes": args.max_episodes, 
    "max_cycles":50, 
    "logs_dir": os.path.join(args.log_dir, log_name), 
    "update_timestep": 30, 
    "save_model_freq": 50_000, 
    "action_std_decay_rate": 0.05, 
    "action_std_decay_freq": int(2.5e5), 
    "min_action_std": 0.1, 
    "lr_actor": 0.0003, 
    "lr_critic": 0.0003, 
    "gamma": 0.99, 
    "K_epochs": 8, 
    "eps_clip": 0.2, 
    "has_continuous_action_space": False, 
    "action_std": 0.5,
    "action_std_init": 0.6 
}

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed=args.seed 

self_agent_id = 0 
other_agent_id = 1 

state_dim = env.observation_space[0].shape[0]
action_dim = env.action_space[0].n 

other_state_dim = state_dim
other_action_dim = action_dim 
other_agent = PPO(other_state_dim, other_action_dim, hyperparams) 

if args.baseline == "noam": 
    self_state_dim = state_dim 
elif args.baseline == "giam": 
    self_state_dim = state_dim + other_state_dim + other_action_dim 
elif args.baseline == "taam": 
    self_state_dim = state_dim + other_action_dim 
elif args.baseline == "toam": 
    self_state_dim = state_dim + other_state_dim 
self_action_dim = action_dim 
self_agent = PPO(self_state_dim, self_action_dim, hyperparams) 


agents = [self_agent, other_agent] 

log_dir = Path(hyperparams["logs_dir"])
# writer = SummaryWriter(log_dir)
log_dir = Path(hyperparams["logs_dir"])
for i in count(0):
    temp = log_dir/('run{}'.format(i)) 
    if temp.exists():
        pass
    else:
        writer = SummaryWriter(temp)
        log_dir = temp
        break

for i_episode in range(1, hyperparams["max_episodes"]+1): 
    state = env.reset() 

    if (args.baseline == "giam") or (args.baseline == "taam"): 
        prev_a_other = np.zeros(other_action_dim) 
    other_state = state[other_agent_id] 

    if args.baseline == "noam": 
        self_state = state[self_agent_id] 
    elif args.baseline == "giam": 
        self_state = np.append(state[self_agent_id], state[other_agent_id]) 
        self_state = np.append(self_state, prev_a_other)  
    elif args.baseline == "taam": 
        self_state = np.append(state[self_agent_id], prev_a_other)  
    elif args.baseline == "toam": 
        self_state = np.append(state[self_agent_id], state[other_agent_id]) 

    self_ep_reward, other_ep_reward, team_ep_reward = 0, 0, 0 

    for t in range(1, hyperparams["max_cycles"]+1):
        actions = tuple([ self_agent.select_action(self_state), other_agent.select_action(other_state) ]) 
        state, reward, done, info = env.step(actions)
        
        if (args.baseline == "giam") or (args.baseline == "taam"): 
            prev_a_other = np.zeros(other_action_dim) 
            prev_a_other[int(actions[other_agent_id])] = 1. 

        other_state = state[other_agent_id] 
        if args.baseline == "noam": 
            self_state = state[self_agent_id] 
        elif args.baseline == "giam": 
            self_state = np.append(state[self_agent_id], state[other_agent_id]) 
            self_state = np.append(self_state, prev_a_other) 
        elif args.baseline == "taam": 
            self_state = np.append(state[self_agent_id], prev_a_other)  
        elif args.baseline == "toam": 
            self_state = np.append(state[self_agent_id], state[other_agent_id]) 

        # saving reward and is_terminals 
        [ agents[a].buffer.rewards.append(reward[a]) for a in range(hyperparams["n_agents"]) ] 
        [ agents[a].buffer.is_terminals.append(done[a]) for a in range(hyperparams["n_agents"]) ] 

        self_ep_reward += reward[self_agent_id]
        other_ep_reward += reward[other_agent_id] 
        team_ep_reward += sum([reward[a] for a in range(hyperparams["n_agents"])])/hyperparams["n_agents"]  # mean 
        
        if all(done): 
            break 
            
    # update PPO agent
    if i_episode % hyperparams["update_timestep"] == 0:
        [ agent.update() for agent in agents ] 

    print("Episode : {} \t\t Average team reward : {}".format(i_episode, team_ep_reward)) 
    writer.add_scalar("rewards/self_ep_reward", self_ep_reward, i_episode) 
    writer.add_scalar("rewards/other_ep_reward", other_ep_reward, i_episode) 
    writer.add_scalar("rewards/team_ep_reward", team_ep_reward, i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        print("Saving model at episode: ", i_episode) 
        self_agent.save(log_dir/("self_agent.pth")) 
        other_agent.save(log_dir/("other_agent.pth")) 


