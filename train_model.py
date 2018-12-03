from unityagents import UnityEnvironment
import numpy as np
import torch
from component.ou_noise import OUNoise
from ddpg_agent import Agent
from collections import deque
from tqdm import tqdm

import matplotlib.pyplot as plt
import importlib
import csv

env = UnityEnvironment(file_name='./Tennis_Windows_x86_64/Tennis.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)
#agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
#agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))
n_episodes=5000
print_every=20
score_calc_sample=100
scores_deque = deque(maxlen=score_calc_sample)

scores = np.zeros(num_agents)
scores_by_episode = []

def save_scores(data, fieldnames, save_name="scores.csv"):
    np.savetxt(save_name, data, delimiter=',', header=fieldnames)

def moving_average(data, n=score_calc_sample) :
    mas = []
    for i in range(1, len(data)+1):
        ma = np.mean(data[i-n:i])
        mas.append(ma)
    
    return mas

total_episodes = []
for i_episode in tqdm(range(1, n_episodes+1)):
    env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
    t = 1
    states = env_info.vector_observations      # get the current state
    agent.reset() # Reset noise with different inertia
    scores = np.zeros(num_agents)
    last_non_zeros_in_batch = 0
    while True:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        rewards = env_info.rewards                         # get reward (for each agent)
        next_states = env_info.vector_observations         # get next state (for each agent)
        dones = env_info.local_done                        # see if episode finished
        agent.step(states, actions, rewards, next_states, dones, t, i_episode )
        scores += rewards
        states = next_states

        if any(dones):
            break
        
        t += 1
    scores_deque.append(np.max(scores))
    scores_by_episode.append(np.max(scores))

    total_episodes.append(i_episode)
    
    if i_episode % print_every == 0:
        print('\rEpisode {}\tRolling Average: {:.4f}\tScore: {:.2f}\tsteps: {}\t'
            .format(i_episode, np.mean(scores_by_episode[-100:]), np.max(scores), t))
        ma = moving_average(scores_by_episode)
        save_scores(np.array([total_episodes, ma, scores_by_episode]).T, "episode,score")
    
    #torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    #torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    
    if np.mean(scores_deque)>=0.5:
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        ma = moving_average(scores_by_episode)
        save_scores(np.array([total_episodes, list(ma), scores_by_episode]).T, "episode,score")
        break

ma = moving_average(scores_by_episode)
save_scores(np.array([total_episodes, ma, scores_by_episode]).T, "episode,rolling average,score")

