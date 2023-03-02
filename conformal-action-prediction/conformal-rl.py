from pettingzoo.mpe import simple_spread_v2 
from ppo import PPO, RolloutBuffer 
from torch.utils.tensorboard import SummaryWriter 
import torch, numpy as np, argparse, os 
from cam import CAM 
import sys 
sys.setrecursionlimit(10000) 


parser = argparse.ArgumentParser() 
parser.add_argument("--seed", type=int, default=0) 
parser.add_argument("--n_agents", type=int, default=2)
parser.add_argument("--cp_update_timestep", type=int, default=30)
parser.add_argument("--log_dir", type=str, default="./debug_logs/cam-actions")
args = parser.parse_known_args()[0] 

log_name = ["cam_new"]  
log_name.append("n_" + str(args.n_agents)) 
log_name.append("cp_update_timestep_" + str(args.cp_update_timestep)) 
log_name.append("seed_" + str(args.seed)) 
log_name = "--".join(log_name) 

hyperparams = {
    "n_agents": args.n_agents, 
    "max_episodes":30_000,
    "initial_start_episodes":50,
    "max_cycles":25,
    "update_timestep": 30, 
    "cp_update_timestep": args.cp_update_timestep, 
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

ego_state_dim = env.observation_space(env.possible_agents[0]).shape[0] + action_dim 

cam_agent = CAM(state_dim=state_dim, act_dim=action_dim) 
ego_agent = PPO(ego_state_dim, action_dim, hyperparams) 
other_agent = PPO(state_dim, action_dim, hyperparams) 

writer = SummaryWriter(hyperparams["logs_dir"])

# TODO: Add initial start runs (without learning) for (1) CP training, and (2) RL training 
# pseudo_agent = PPO(state_dim, action_dim, hyperparams) 
# TODO: cannot use pseudo agent. give random conformal actions in beginning, or pretrain conformal model on single agent setting maybe. Or have 2 psuedo agents for pre-training. 
for i_episode in range(1, hyperparams["initial_start_episodes"]+1): 
    state = env.reset() 
    for t in range(1, hyperparams["max_cycles"]+1): 
        action = {} 
        padded_S_pseudo = np.zeros(action_dim)
        action[env.possible_agents[0]] = ego_agent.select_action(np.concatenate([state[env.possible_agents[0]], padded_S_pseudo])) 
        action[env.possible_agents[1]] = other_agent.select_action(state[env.possible_agents[1]]) 
        state, reward, done, is_terminals, info = env.step(action) 
        cam_agent.buffer.states.append(state[env.possible_agents[1]])
        cam_agent.buffer.actions.append(action[env.possible_agents[1]]) 
cam_agent.create_cp_dataset() 
cam_agent.train_cp_model() 

cam_agent.buffer.clear() 
ego_agent.buffer.clear() 
other_agent.buffer.clear() 

for i_episode in range(1, hyperparams["max_episodes"]+1): 
    state = env.reset() 
    ep_reward = 0 
    for t in range(1, hyperparams["max_cycles"]+1):
        action = {}
        # Prepare state for ego agent 
        # calculate padded conformal set S of other_agent's actions; concatenate them to ego agent obs 
        output, S = cam_agent.get_conformal_action_predictions(ego_agent.batchify_obs(state, agent_id=1).reshape(1, -1)) #TODO: Remove hardcoding of shape 
        padded_S = np.pad(S[0], (0, action_dim-S[0].shape[0])) 
        action[env.possible_agents[0]] = ego_agent.select_action(np.concatenate([state[env.possible_agents[0]], padded_S])) 
        action[env.possible_agents[1]] = other_agent.select_action(state[env.possible_agents[1]]) 
        state, reward, done, is_terminals, info = env.step(action) 
        # saving other agent's states and actions for conformal prediction 
        cam_agent.buffer.states.append(state[env.possible_agents[1]])
        cam_agent.buffer.actions.append(action[env.possible_agents[1]]) 
        # saving reward and is_terminals
        ego_agent.buffer.rewards.append(reward[env.possible_agents[0]])
        ego_agent.buffer.is_terminals.append(done[env.possible_agents[0]])
        other_agent.buffer.rewards.append(reward[env.possible_agents[1]])
        other_agent.buffer.is_terminals.append(done[env.possible_agents[1]])

        ep_reward += sum([reward[env.possible_agents[a]] for a in range(hyperparams["n_agents"])])/hyperparams["n_agents"]  # mean 
        # ep_reward += reward[env.possible_agents[0]] # tracking only ego agent's rewards 

        if any(list(done.values())): 
            break 
            
    # update PPO agent
    if i_episode % hyperparams["cp_update_timestep"] == 0: 
        # cam agent to create dataset of experiences and train cp 
        cam_agent.create_cp_dataset() 
        cp_loss, cp_acc =  cam_agent.train_cp_model() 
        writer.add_scalar("cp/model_loss", cp_loss, i_episode) 
        writer.add_scalar("cp/model_acc", cp_acc, i_episode) 
        top1_avg, top5_avg, coverage_avg, size_avg = cam_agent.validate_model() 
        writer.add_scalar("cp/top1_avg", top1_avg, i_episode) 
        writer.add_scalar("cp/top5_avg", top5_avg, i_episode) 
        writer.add_scalar("cp/coverage_avg", coverage_avg, i_episode) 
        writer.add_scalar("cp/size_avg", size_avg, i_episode) 

    # update PPO agent
    if i_episode % hyperparams["update_timestep"] == 0: 
        ego_agent.update() 
        other_agent.update() 

    print("Episode : {} \t\t Average Reward : {}".format(i_episode, ep_reward)) 
    writer.add_scalar("Episodic Return", ep_reward, i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        print("Saving model at episode: ", i_episode) 
        ego_agent.save(hyperparams["logs_dir"] + "/ego_agent.pth" ) 
        other_agent.save(hyperparams["logs_dir"] + "/other_agent.pth" ) 

