from ddqn import Agent
import numpy as np
import gym
import matplotlib.pyplot as plt

#set up gym environment for lander
env = gym.make('LunarLander-v2')
# we will run for 401 episodes so that we can see the very last episodes
n_games = 401
#initialize agent for this environment, set variables
agent = Agent(alpha=.001, gamma=.99, n_actions=4,batch_size=64, input_dims=[8], mem_size=100000, 
 fc1_dims=128, fc2_dims=128, replace=100)
 # empty lists for some data tracking
scores, l_ep = [], []
#main loop for each episode, sets some values
for episode in range(n_games):
    done = False
    score = 0
    observation = env.reset()
    # main loop for actions/updating training while in an episode
    while not done:
        # only render every 10 episodes
        if episode%10 == 0:
            env.render()
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
    scores.append(score)
    l_ep.append(episode)
    #rolling average score in output
    avg_score = np.mean(scores[-20:])

    print(f'episode {episode}', f'score {round(score, 2)}', f'rolling avg {round(avg_score, 2)}', f'epsilon {round(agent.epsilon, 2)}')
    
# create plot at the end to show scores over time
plt.scatter(l_ep, scores)
plt.ylabel('scores')
plt.show()
env.close()
