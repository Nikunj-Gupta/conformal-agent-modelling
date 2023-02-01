import numpy as np, torch 


class RandomPolicy:
    def __init__(self, action_dim, has_continuous_action_space=False): 
        self.action_dim = action_dim 
        self.has_continuous_action_space = has_continuous_action_space 
    def select_action(self, state=None): 
        if self.has_continuous_action_space:
            return np.random.uniform(0, 1, size=self.action_dim) 
        else: 
            return np.random.randint(1, self.action_dim) 
    def batchify_obs(self, obs, device): 
        """Converts PZ style observations to batch of torch arrays."""
        # convert to list of np arrays
        obs = np.stack([obs[a] for a in obs], axis=0)
        # transpose to be (batch, channel, height, width)
        obs = obs.transpose(0, -1, 1, 2)
        # convert to torch
        obs = torch.FloatTensor(obs).to(device) 
        return obs 


