from pettingzoo.mpe import simple_spread_v2 

env = simple_spread_v2.parallel_env(N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False) 

state = env.reset() 
for i in range(1): 
    action = {}
    for a in range(2): 
        action["agent_"+str(a)] = env.action_space(env.possible_agents[a]).sample() 
    state, reward, done, is_terminals, info = env.step(action) 

    print(state, reward, done, is_terminals, info )

