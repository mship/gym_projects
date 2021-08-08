import gym
import numpy as np
# set up environment - reset to make sure it is a clean slate
env = gym.make('MountainCar-v0')

env.reset()

#set variables for q learning formula
alpha = 0.25
gamma = 0.85
episodes = 15000
# only render some episodes - speeds it up!
show_every = 500 
# this is where we get the size of our q table. 
os_size = [20] * len(env.observation_space.high)
os_win_size = (env.observation_space.high - env.observation_space.low) / os_size
# set up q table with initially random values
q_table = np.random.uniform(low = -2, high=0, size = (os_size + [env.action_space.n]))
# change continuous states to discrete states
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / os_win_size
    return tuple(discrete_state.astype(np.int))

# main loop for q learning
for episode in range(episodes):
    print(episode)
    discrete_state = get_discrete_state(env.reset())
    # done is a variable used to keep the while loop running
    done = False
    
    while(not done):
        # only show some episodes
        if(episode % show_every == 0):
            env.render()
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        
        if(not done):
            # look for best value of q, then find our current q value, then find new q based on formula
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0
        discrete_state = new_discrete_state

env.close()

