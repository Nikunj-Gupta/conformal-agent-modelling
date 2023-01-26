import numpy as np 


class RandomPolicy:
    def __init__(self, action_dim, hyperparams): 
        self.action_dim = action_dim 
        self.has_continuous_action_space = hyperparams["has_continuous_action_space"] 
    def select_action(self, state=None): 
        if self.has_continuous_action_space:
            return np.random.uniform(0, 1, size=self.action_dim) 
        else: 
            return np.random.randint(1, self.action_dim) 
    def batchify_obs(self, obs, agent_id): 
        return None 

