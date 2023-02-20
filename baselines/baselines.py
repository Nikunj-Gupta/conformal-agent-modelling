from pettingzoo.mpe import simple_spread_v2, simple_push_v2, simple_speaker_listener_v3, simple_world_comm_v2, simple_tag_v2, simple_adversary_v2 
from ppo import PPO 
from torch.utils.tensorboard import SummaryWriter 
import torch, numpy as np, argparse, os 
from itertools import count
from pathlib import Path 
from collections import defaultdict 

parser = argparse.ArgumentParser() 
parser.add_argument("--envname", type=str) 
parser.add_argument("--baseline", type=str) 

parser.add_argument("--n_agents", type=int, default=2) 
parser.add_argument("--n_adversaries", type=int, default=4) 
parser.add_argument("--max_episodes", type=int, default=100_000) 
parser.add_argument("--max_cycles", type=int, default=25) 
parser.add_argument("--update_timestep", type=int, default=30) 
parser.add_argument("--save_model_freq", type=int, default=5_000) 

parser.add_argument("--seed", type=int, default=0) 
parser.add_argument("--log_dir", type=str, default="./env-search-new") 

args = parser.parse_known_args()[0] 

log_name = [
    args.baseline, 
    args.envname, 
    "n_" + str(args.n_agents)
] 
log_name.append("seed_" + str(args.seed)) 
log_name = "--".join(log_name) 

hyperparams = {
    # ENV / EXP hyperparams 
    "n_agents": args.n_agents, 
    "n_adversaries": args.n_adversaries, 
    "max_episodes": args.max_episodes, 
    "max_cycles": args.max_cycles, 
    "update_timestep": args.update_timestep, 
    "save_model_freq": args.save_model_freq, 
    "logs_dir": os.path.join(args.log_dir, log_name), 
    # PPO hyperparams 
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

lead_adversary_id = None 
adversary_ids = None 
self_agent_id = None 
other_agent_id = None 

if args.envname == "simple_spread_v2": 
    env = simple_spread_v2.parallel_env(
        N=hyperparams["n_agents"], 
        continuous_actions=hyperparams["has_continuous_action_space"] 
    )
    print(env.possible_agents) 
    self_agent_id = 0 
    other_agent_id = 1 

if args.envname == "simple_push_v2": 
    env = simple_push_v2.parallel_env(continuous_actions=hyperparams["has_continuous_action_space"])
    print(env.possible_agents) 
    self_agent_id = 1 
    other_agent_id = 0 

if args.envname == "simple_speaker_listener_v3": 
    env = simple_speaker_listener_v3.parallel_env(continuous_actions=hyperparams["has_continuous_action_space"])
    print(env.possible_agents) 
    self_agent_id = 1 
    other_agent_id = 0

if args.envname == "simple_world_comm_v2": 
    env = simple_world_comm_v2.parallel_env(
        num_good=hyperparams["n_agents"], 
        num_adversaries=hyperparams["n_adversaries"], 
        continuous_actions=hyperparams["has_continuous_action_space"]
    ) 
    print(env.possible_agents) 
    lead_adversary_id = 0  
    adversary_ids = list(range(1, hyperparams["n_adversaries"])) 
    self_agent_id = hyperparams["n_adversaries"] - 1 + 1  
    other_agent_id = hyperparams["n_adversaries"] - 1 + 2 

if args.envname == "simple_adversary_v2": 
    env = simple_adversary_v2.parallel_env(
        N=hyperparams["n_agents"], 
        continuous_actions=hyperparams["has_continuous_action_space"]
    ) 
    print(env.possible_agents) 
    adversary_ids = [0] 
    self_agent_id = 1
    other_agent_id = 2 

if args.envname == "simple_tag_v2": 
    env = simple_tag_v2.parallel_env(
        num_good=hyperparams["n_agents"], 
        num_adversaries=hyperparams["n_adversaries"], 
        continuous_actions=hyperparams["has_continuous_action_space"]
    ) 
    print(env.possible_agents) 
    adversary_ids = list(range(hyperparams["n_adversaries"])) 
    self_agent_id = hyperparams["n_adversaries"] - 1 + 1 
    other_agent_id = hyperparams["n_adversaries"] - 1 + 2 

# seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.reset(seed=args.seed) 

# other agent's state and action dimensions 
other_state_dim = env.observation_space(env.possible_agents[other_agent_id]).shape[0]
other_action_dim = env.action_space(env.possible_agents[other_agent_id]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[other_agent_id]).n 
other_agent = PPO(other_state_dim, other_action_dim, hyperparams)   # Other agent 

# lead adversary agent's state and action dimensions 
if lead_adversary_id!=None: 
    lead_adversary_state_dim = env.observation_space(env.possible_agents[lead_adversary_id]).shape[0]
    lead_adversary_action_dim = env.action_space(env.possible_agents[lead_adversary_id]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[lead_adversary_id]).n 
    lead_adversary_agent = PPO(lead_adversary_state_dim, lead_adversary_action_dim, hyperparams)   # Other agent 

# adversary agent's state and action dimensions 
if adversary_ids!=None: 
    adversary_state_dim = env.observation_space(env.possible_agents[adversary_ids[0]]).shape[0]
    adversary_action_dim = env.action_space(env.possible_agents[adversary_ids[0]]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[adversary_ids[0]]).n 
    adversary_agents = [PPO(adversary_state_dim, adversary_action_dim, hyperparams) for _ in adversary_ids] # Adversary agents 
# self agent's state and action dimensions based on baseline 

if args.baseline == "noam": 
    self_state_dim = env.observation_space(env.possible_agents[self_agent_id]).shape[0]
    self_action_dim = env.action_space(env.possible_agents[self_agent_id]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[self_agent_id]).n 
if args.baseline == "giam": 
    # current O_self + current O_other + previous a_other (one hot vector) --> a_self
    self_state_dim = env.observation_space(env.possible_agents[self_agent_id]).shape[0] + other_state_dim + other_action_dim 
    self_action_dim = env.action_space(env.possible_agents[self_agent_id]).shape[0] if hyperparams["has_continuous_action_space"] else env.action_space(env.possible_agents[self_agent_id]).n 
self_agent = PPO(self_state_dim, self_action_dim, hyperparams)      # Ego/Self agent 


log_dir = Path(hyperparams["logs_dir"])
writer = SummaryWriter(log_dir)

# log_dir = Path(hyperparams["logs_dir"])
# for i in count(0):
#     temp = log_dir/('run{}'.format(i)) 
#     if temp.exists():
#         pass
#     else:
#         writer = SummaryWriter(temp)
#         log_dir = temp
#         break

for i_episode in range(1, hyperparams["max_episodes"]+1): 
    state = env.reset() 
    ep_reward = 0 
    all_rewards = defaultdict(float) 

    if args.baseline == "giam": prev_a_other = np.zeros(other_action_dim) # zero vector as initial value for prev_a_other 

    for t in range(1, hyperparams["max_cycles"]+1):
        action = {}
        if args.baseline == "noam": self_state = state[env.possible_agents[self_agent_id]] 
        if args.baseline == "giam": 
            self_state = np.append(state[env.possible_agents[self_agent_id]], state[env.possible_agents[other_agent_id]]) 
            self_state = np.append(self_state, prev_a_other)  
        action[env.possible_agents[self_agent_id]] = self_agent.select_action(self_state) 
        action[env.possible_agents[other_agent_id]] = other_agent.select_action(state[env.possible_agents[other_agent_id]]) 
        if lead_adversary_id!=None: action[env.possible_agents[lead_adversary_id]] = lead_adversary_agent.select_action(state[env.possible_agents[lead_adversary_id]]) 
        if adversary_ids!=None: 
            for i, a in enumerate(adversary_ids): 
                action[env.possible_agents[a]] = adversary_agents[i].select_action(state[env.possible_agents[a]]) 

        if args.baseline == "giam": 
            prev_a_other = np.zeros(other_action_dim) 
            prev_a_other[int(action[env.possible_agents[other_agent_id]])] = 1. 

        state, reward, done, is_terminals, info = env.step(action) 

        # saving reward and is_terminals
        self_agent.buffer.rewards.append(reward[env.possible_agents[self_agent_id]])
        self_agent.buffer.is_terminals.append(done[env.possible_agents[self_agent_id]])
        other_agent.buffer.rewards.append(reward[env.possible_agents[other_agent_id]])
        other_agent.buffer.is_terminals.append(done[env.possible_agents[other_agent_id]])
        if lead_adversary_id!=None: 
            lead_adversary_agent.buffer.rewards.append(reward[env.possible_agents[lead_adversary_id]])
            lead_adversary_agent.buffer.is_terminals.append(done[env.possible_agents[lead_adversary_id]])
        if adversary_ids!=None: 
            for i, a in enumerate(adversary_ids): 
                adversary_agents[i].buffer.rewards.append(reward[env.possible_agents[a]])
                adversary_agents[i].buffer.is_terminals.append(done[env.possible_agents[a]])
       
        # ep_reward += sum([reward[env.possible_agents[a]] for a in range(hyperparams["n_agents"])])/hyperparams["n_agents"]  # mean 
        ep_reward += reward[env.possible_agents[self_agent_id]] # tracking only ego agent's rewards 
        for a in env.possible_agents: all_rewards[a] += reward[a] # tracking all agents' rewards 

        # if continuous action space; then decay action std of ouput action distribution
        if hyperparams["has_continuous_action_space"] and t % hyperparams["action_std_decay_freq"] == 0:
            self_agent.decay_action_std(hyperparams["action_std_decay_rate"], hyperparams["min_action_std"]) 
            other_agent.decay_action_std(hyperparams["action_std_decay_rate"], hyperparams["min_action_std"]) 
            if lead_adversary_id!=None: 
                lead_adversary_agent.decay_action_std(hyperparams["action_std_decay_rate"], hyperparams["min_action_std"]) 
            if adversary_ids!=None: 
                for i, a in enumerate(adversary_ids): 
                    adversary_agents[i].decay_action_std(hyperparams["action_std_decay_rate"], hyperparams["min_action_std"]) 

        if all(list(done.values())): 
            break 
    
    # update PPO agent
    if i_episode % hyperparams["update_timestep"] == 0:
        self_agent.update() 
        other_agent.update() 
        if lead_adversary_id!=None: 
            lead_adversary_agent.update() 
        if adversary_ids!=None: 
            for i, a in enumerate(adversary_ids): 
                adversary_agents[i].update() 

    print("Episode : {} \t\t Self agent reward : {}".format(i_episode, ep_reward)) 
    writer.add_scalar("self_reward", ep_reward, i_episode) 
    for a in env.possible_agents: 
        writer.add_scalar(a+"_reward", all_rewards[a], i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        print("Saving model at episode: ", i_episode) 
        self_agent.save(log_dir/("self_agent.pth")) 
        other_agent.save(log_dir/("other_agent.pth")) 
        if lead_adversary_id!=None: 
            lead_adversary_agent.save(log_dir/("lead_adversary_agent.pth")) 
        if adversary_ids!=None: 
            for i, a in enumerate(adversary_ids): 
                adversary_agents[i].save(log_dir/("adversary_agent_"+str(i)+".pth")) 
        