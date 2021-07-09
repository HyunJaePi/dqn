#!/usr/bin/env python
# coding: utf-8

# # DQN -- CartPole ver0.9
# v0.8: a bug fixed (deque out of index )  
# v0.9: save/load a trained model

# ### Note to Erchi  (2021Apr04)
# I implemented saving a model and retraining it from saved files.  
#   
# I will do this for DQN_Atari. This way, we will not only have checkpoints (e.g. every 500 episodes) but also you and I can continue to train the model that the other person previously trained. Once I finish coding, I will update you again. Meantime, you can look at this code and see what is going on.  
#   
# HW3 will be released soon. Then let's finish HW3 first and continue to work on this project.
# 

# In[14]:


import gym
import numpy as np
import torch
from torch import nn
from collections import namedtuple, deque
from copy import copy, deepcopy
from PIL import Image
import sys
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import os
from pathlib import Path

import pickle


# ### 1. define classes

# In[15]:


# class ReplayMemory
class ReplayMemory:
    
    def __init__(self, memory_capacity=100000, batch_size=32, replay_threshold=0.25):
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.n_memories = 0
        self.replay_threshold = replay_threshold
        #experience = tuple() # (state, action, reward, next_state, done)
        self.replay_memory = deque(maxlen=self.memory_capacity) # a circular queue, store experiences 
        
    
    def append(self, experience):
        self.replay_memory.append(experience)
        self.n_memories += 1 if self.n_memories < self.memory_capacity else 0 

    
    def sample_batch(self):
        try:
            indices = np.random.choice(self.n_memories, self.batch_size)   
        except:
            print(f'{sys.exc_info()[0]} occurred. Replay memory may not be filled enough.')
            
        # debugging
        try:
            samples = [self.replay_memory[i] for i in indices] 
        except:
            print(f'\nindice: \n {indices}, \nn_memories: {self.n_memories}')
            
        return samples
                
    
    def init_random_fill(self, env):
        '''
        fill 1/4 of replay memory with random experiences; needed for sample_batch()
        '''
        while self.n_memories/self.memory_capacity < self.replay_threshold:
            state = env.reset()    
            action = env.action_space.sample() # random action
            next_state, reward, done, _ = env.step(action) 
            experience = (state, action, reward, next_state, done)
            self.append(experience)
    
    


        


# In[16]:


# class DQN

class DQN(nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQN, self).__init__()
        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = env.action_space.n
        self.learning_rate = learning_rate
        self.device = device       
                
        # network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, 16, bias=True),
            nn.ReLU(),
            nn.Linear(16, self.n_outputs, bias=True)
        )
        
        # device
        if self.device == 'cuda':
            self.network.to('cuda')
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        
    def get_q_values(self, state):
        '''
        input: state (tensor)
        output: action values (q values)
        ''' 
        state = torch.FloatTensor(state).to(device=self.device) if not torch.is_tensor(state) else state
        return self.network(state)
    
    
    # save weights
    def save(self, filename=None):
        if filename is None:
            path_saved = Path(r'./saved')
            filename = os.path.join(path_saved,'model.pt')
        #torch.save(self.state_dict(), filename)
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename)
    
    
    # load weights
    def load(self, filename=None):
        if filename is None:
            path_saved = Path(r'./saved')
            filename = os.path.join(path_saved,'model.pt')
        try: 
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict']) # load a model
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("loading error --- something wrong")
            


# In[17]:


# class Agent
class Agent:
    
    def __init__(self, network, env=None, replay_memory=None, continue_training=False):
        self.network = network
        self.target_network = deepcopy(network) # target network; updated occasionally
        #csv lists for saving data for plotting
        self.rewards_csv=[]
        self.episodes_csv=[]
        #if not continue_training: # new training
        self.env = env
        self.replay_memory = replay_memory
        self._initialize()

    
    def _initialize(self, max_episodes=2500, epsilon=0.1, gamma=0.99):    
        self.epsilon = epsilon # for epsilon-greedy
        self.gamma = gamma # discount factor
        
        self.state = self.env.reset()
        self.next_state = []
        
        self.rewards = 0 # total rewards collected in a single episode
        self.rewards_list = [] # a list of the total reward collected on each episodes
        self.mean_rewards_list = [] # to plot a smooth curve of rewards
        self.reward_th = 195
        self.window = 100 # window size for moving average
        
        self.step_count = 0
        self.episode_count = 0
        self.max_episodes = max_episodes
        
        self.episode_loss = []
        self.loss_list = [] # a list of the mean loss on each episode
        
    def set_network(self, network):
        self.network = network
        #self.target_network = deepcopy(network) # target network; updated occasionally
        #self.replay_memory = replay_memory
        
        
    def set_env(self, env):
        self.env = env
        
    
    def take_step(self):
        '''
        by taking one time-step -- experience, update replay memory & the current state
        '''
        action = self.epsilon_greedy_action()    
        self.next_state, reward, done, _ = self.env.step(action) # _ = info
        experience = (self.state, action, reward, self.next_state, done)
        self.replay_memory.append(experience)
        
        self.rewards += reward 
        self.state = self.next_state.copy()

        return done
   

    def epsilon_greedy_action(self, epsilon=0.1):
        self.epsilon = epsilon
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # random action
        else:
            q_values = self.network.get_q_values(self.state)
            action = torch.max(q_values, dim=-1)[1].item() # torch.max returns values(=[0]) & indices(=[1] = action value)
        
        return action


    def calculate_loss(self):
        experiences = self.replay_memory.sample_batch()    
        states, actions, rewards, next_states, dones = zip(*[e for e in experiences])
                
        # convert them to tensors
        states_t = torch.FloatTensor(states).to(device=self.network.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(device=self.network.device) # unsqueeze -> [32] -> [32,1]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device).bool() # boolean
        next_states_t = torch.FloatTensor(next_states).to(device=self.network.device)
        
        # max_q_next: n_data x n_actions (32 x 2) -> max() -> maximun action values -> torch.max returns a tuple (values, indices) -> [0] takes values only -> needs to be detached
        max_q_next = torch.max(self.target_network.get_q_values(next_states_t), dim=1)[0].detach()
        max_q_next[dones_t] = 0 # terminal states don't have the 'next' q values; see algorithm 1 in Mnih et al. 2013
        y = torch.unsqueeze(rewards_t + self.gamma*max_q_next, 1)
        # q_curr: 32 x 2 -> gather values according to action taken (actions_t); no detach??
        q = torch.gather(self.gamma*self.network.get_q_values(states_t), 1, actions_t)
        loss = nn.MSELoss()
        
        return loss(q, y)
    

    def update_network(self):
        # optim: zero_grad() -> loss -> backward() -> step()
        self.network.optimizer.zero_grad()
        loss = self.calculate_loss()
        loss.backward()
        self.network.optimizer.step()
        
        if self.network.device == 'cuda':
            self.episode_loss.append(loss.detach().cpu().numpy())
        else:
            self.episode_loss.append(loss.detach().numpy())
        
    
    def sync_network(self):
        # update params of the target_network
        curr_state_dict = self.network.state_dict() # state_dict from the current network
        self.target_network.load_state_dict(curr_state_dict) # update target_network
        
        
    def train(self, network_update_freq=4, network_sync_freq=5000):
        while True:            
            # reset before each episode starts
            self.rewards = 0
            self.episode_loss = []
            self.state = env.reset() 
            done = False
            
            while not done:
                # 1. experience, update replay memory, state & rewards
                done = self.take_step() 
                # 2. calculate loss, update & sync networks
                if self.step_count%network_update_freq == 0:
                    self.update_network() 
                if self.step_count%network_sync_freq == 0:    
                    self.sync_network() 
                    
                self.step_count += 1
            
            # store values for visualization
            self.loss_list.append(np.mean(self.episode_loss))
            self.rewards_list.append(self.rewards)
            self.mean_rewards_list.append(np.mean(self.rewards_list[-self.window:]))
            
            # monitor progress
            print(f'\rEpisode: {self.episode_count}, Step: {self.step_count}, Loss: {self.loss_list[-1]}, Rewards: {self.rewards_list[-1]}', end='')
            
            self.episode_count += 1 # increase episode_count
            #append numbers
            if self.episode_count%100==0:
                self.episodes_csv.append(self.episode_count)
                self.rewards_csv.append(self.rewards_list[-1])
            
            if self.mean_rewards_list[-1] > self.reward_th or self.episode_count > self.max_episodes:
                print(f'\nTraining done')
                break 
            
            # for testing 'save' and continuing training
            '''
            if (self.episode_count != 0) and (self.episode_count%500 == 0):
                break
            '''
        dataframe = pd.DataFrame({'rewards':self.rewards_csv,'episodes':self.episodes_csv})
        dataframe.to_csv("gam097.csv",index=False,sep=',')



                

    
    # save mean rewards
    def save_mean_rewards(self, filename):        
        file = open(filename, 'w') # new file
        for r in self.mean_rewards_list:
            file.write('{}\n'.format(r))
        file.close()


# In[18]:


# class Units -- TODO

# Utils
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        
def load_object(filename):
    with open(filename, 'rb') as input:
        output = pickle.load(input)
        return output

    


# ### 2. Simulation

# In[19]:


# simulation

# initialization
env = gym.make('CartPole-v0')
net = DQN(env)
mem = ReplayMemory()
agnt = Agent(net, env, mem)

# fill_replay_memory
mem.init_random_fill(env)

# train
agnt.train()


# In[13]:


# plot
plt.figure(1, figsize=(8,4))
plt.plot(agnt.rewards_list, label='reward_episode')
plt.plot(agnt.mean_rewards_list, label='reward_episode')
plt.xlabel('Episodes', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.ylim([0, np.round(agnt.reward_th)*1.05])
path_outputs = os.getcwd()
plt.savefig(os.path.join(path_outputs, 'rewards.png'))
plt.show()

# TODO: plot loss vs. episode here


# In[37]:


plt.close()


# ### 3. Save a trained model

# In[21]:


# save
path_saved = os.getcwd()

# model
fname_model = os.path.join(path_saved,'model.pt')
net.save(filename=fname_model)

# replay memory
fname_mem= os.path.join(path_saved,'replay_memory.pkl')
save_object(mem, fname_mem)

# agent
fname_agnt= os.path.join(path_saved,'agent.pkl')
save_object(agnt, fname_agnt)


# rewards
# fname_rewards = os.path.join(path_saved,'mean_rewards_list.txt')
# agnt.save_mean_rewards(filename=fname_rewards)


# ### 4. Load a model

# In[22]:


# delete objects
del agnt, net, mem
env.close()


# In[23]:


# load DQN
net = DQN(env)
path_saved = os.getcwd()
fname_model = os.path.join(path_saved,'model.pt')
net.load(fname_model)

# load replay memory
fname_mem = os.path.join(path_saved,'replay_memory.pkl')
mem = load_object(fname_mem)

# load agent
fname_agnt = os.path.join(path_saved,'agent.pkl')
agnt = load_object(fname_agnt)


# In[24]:


env = gym.make('CartPole-v0')
agnt.set_network(net)
agnt.set_env(env)


# ### 5. continue to train a model

# In[29]:


agnt.train()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### 3. Display a trained CartPole

# In[7]:


#Run the env
env = gym.make('CartPole-v0')
frames=[]
agnt.state= env.reset()
for i in range(300):
    frames.append(env.render(mode='rgb_array'))
    action = agnt.epsilon_greedy_action(0) # always greedy action
    agnt.state, reward, done, _ = env.step(action)
    print(f'\rframe: {i+1}, action: {action}', end='')
    if done:
        break
    
env.close()


# ### 4. Save it as a gif file
# https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
# 
# TODO: fix a warning message
# 

# In[8]:


from matplotlib import animation
import matplotlib.pyplot as plt
import gym 

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='animation_cartpole.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 100.0, frames[0].shape[0] / 100.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='pillow', fps=60)


#Run the env
env = gym.make('CartPole-v0')
frames=[]
agnt.state= env.reset()
for i in range(300):
    frames.append(env.render(mode='rgb_array'))
    action = agnt.epsilon_greedy_action(0) # always greedy action
    agnt.state, reward, done, _ = env.step(action)
    print(f'\rframe: {i+1}, action: {action}', end='')
    if done:
        break
    
env.close()
save_frames_as_gif(frames)


# In[ ]:





# In[ ]:





# In[ ]:




