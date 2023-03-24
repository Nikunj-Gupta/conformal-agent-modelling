import lbforaging, gym
from ppo import PPO  
from torch.utils.tensorboard import SummaryWriter 
import torch, numpy as np, argparse, os 
from cam import CAM 
import sys 
sys.setrecursionlimit(10000) 


parser = argparse.ArgumentParser() 
parser.add_argument("--env", type=str, default="Foraging-12x12-2p-4f-coop-v1") 
parser.add_argument("--max_episodes", type=int, default=500_000) 
parser.add_argument("--cp_update_timestep", type=int, default=30)
parser.add_argument("--seed", type=int, default=0) 
parser.add_argument("--log_dir", type=str, default="./debug_logs/lbf-exact-action")
args = parser.parse_known_args()[0] 

log_name = ["lbf-exact-action"] 
log_name.append("cp_update_timestep_" + str(args.cp_update_timestep)) 
log_name.append("seed_" + str(args.seed)) 
log_name = "--".join(log_name) 


# env = gym.make("Foraging-8x8-2p-1f-coop-v1")
env = gym.make(args.env)
env.reset() 

hyperparams = {
    "n_agents": env.n_agents, 
    "max_episodes":args.max_episodes,
    "initial_start_episodes":50,
    "max_cycles":50, 
    "update_timestep": 30, 
    "cp_update_timestep": args.cp_update_timestep, 
    "save_model_freq": 50_000, 
    "logs_dir": os.path.join(args.log_dir, log_name), 
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

ego_state_dim = state_dim + state_dim + 1 # action_dim 

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
        padded_S_pseudo = np.zeros(1) 
        actions = tuple([ ego_agent.select_action(np.concatenate([
                state[self_agent_id], 
                state[other_agent_id], 
                padded_S_pseudo]
            )), other_agent.select_action(state[other_agent_id]) ]) 
        state, reward, done, info = env.step(actions)

        cam_agent.buffer.states.append(state[other_agent_id])
        cam_agent.buffer.actions.append(actions[other_agent_id]) 
cam_agent.create_cp_dataset() 
cam_agent.train_cp_model() 

cam_agent.buffer.clear() 
ego_agent.buffer.clear() 
other_agent.buffer.clear() 

for i_episode in range(1, hyperparams["max_episodes"]+1): 
    state = env.reset() 
    self_ep_reward, other_ep_reward, team_ep_reward = 0, 0, 0 
    ep_reward = 0 
    for t in range(1, hyperparams["max_cycles"]+1):
        action = {}
        # Prepare state for ego agent 
        # calculate padded conformal set S of other_agent's actions; concatenate them to ego agent obs 
        output = cam_agent.get_exact_action_prediction(ego_agent.batchify_obs(state, agent_id=1).reshape(1, -1))
        padded_S = np.array([output]) 
        
        actions = tuple([ ego_agent.select_action(np.concatenate([
        state[self_agent_id], 
        state[other_agent_id], 
        padded_S]
            )), other_agent.select_action(state[other_agent_id]) ]) 
        state, reward, done, info = env.step(actions)

        # saving other agent's states and actions for conformal prediction 
        cam_agent.buffer.states.append(state[other_agent_id])
        cam_agent.buffer.actions.append(actions[other_agent_id]) 
        # saving reward and is_terminals
        ego_agent.buffer.rewards.append(reward[self_agent_id])
        ego_agent.buffer.is_terminals.append(done[self_agent_id])
        other_agent.buffer.rewards.append(reward[other_agent_id])
        other_agent.buffer.is_terminals.append(done[other_agent_id])

        self_ep_reward += reward[self_agent_id]
        other_ep_reward += reward[other_agent_id] 
        team_ep_reward += sum([reward[a] for a in range(hyperparams["n_agents"])])/hyperparams["n_agents"]  # mean 

        if all(done): 
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

    print("Episode : {} \t\t Average team reward : {}".format(i_episode, team_ep_reward)) 
    writer.add_scalar("rewards/self_ep_reward", self_ep_reward, i_episode) 
    writer.add_scalar("rewards/other_ep_reward", other_ep_reward, i_episode) 
    writer.add_scalar("rewards/team_ep_reward", team_ep_reward, i_episode) 
    if (i_episode % hyperparams["save_model_freq"])==0: 
        print("Saving model at episode: ", i_episode) 
        ego_agent.save(hyperparams["logs_dir"] + "/ego_agent.pth" ) 
        other_agent.save(hyperparams["logs_dir"] + "/other_agent.pth" ) 

