# pong_v12.py
#
# v1.0 -- working but may need to be  more optimized
# v1.1 -- 2 cnn layers
# v1.2 -- ResNet


import gym
import numpy as np
import torch
from torch import nn
from collections import namedtuple, deque
from copy import copy, deepcopy
from PIL import Image
import sys
import matplotlib.pyplot as plt
from torch.nn import Conv2d, Linear, Sequential, BatchNorm2d, LeakyReLU
import time
import sys
import os
from pathlib import Path
import pickle
import gc
import torch.nn.functional as F
import random


# params
RETRAINING = False
#PATH = '/work/asidasi/pong_v12/' # you need to create a folder first
PATH = './'

ENVIRONMENT = 'PongDeterministic-v4'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")    

# critical parameters
GAMMA = 0.97 # discount rate
MEMORY_CAPACITY = 50000
REPLAY_MEM_TH = 0.8 # 80% of the total memory will be filled
NET_SYNC_FREQ = 10000 # every 10000 steps

IMAGE_SIZE = (64, 80) # (width, height) -- for Pong
BATCH_SIZE = 64
LEARNING_RATE = 0.00025
EPSILON_MIN = 0.05
SAVE_FREQ = 200 # every 100 episodes
NET_UPDATE_FREQ = 1 # Deterministic-v4 already skips 4 frames
MAX_EPISODE = 5000
REWARD_TH = 18 # pong: 18, others: 300, cartpole:195

RESNET = True

# class DDQN
# dueling DQN
# https://medium.com/@parsa_h_m/deep-reinforcement-learning-dqn-double-dqn-dueling-dqn-noisy-dqn-and-dqn-with-prioritized-551f621a9823

class DDQN(nn.Module):
    def __init__(self, n_outputs):
        super(DDQN, self).__init__()
        self.n_inputs = 4; # 4 frames #cartpole--env.observation_space.shape[0]
        self.n_outputs = n_outputs #env.action_space.n
        self.learning_rate = LEARNING_RATE
        self.device = DEVICE
                
        # layers
        self.conv1 = Conv2d(self.n_inputs, 32, kernel_size=8, stride=4)
        self.bn1 = BatchNorm2d(32)
        w_next, h_next = self.cal_conv2d_image_size(IMAGE_SIZE[0],IMAGE_SIZE[1], kernel_size=8, stride=4)
        
        if RESNET:
            self.conv2 = Conv2d(32, 32, kernel_size=(1,1), stride=1) 
            self.bn2 = BatchNorm2d(32)
            w_next, h_next = self.cal_conv2d_image_size(w_next, h_next, kernel_size=1, stride=1)
            self.conv3 = Conv2d(32, 32, kernel_size=(1,1), stride=1) 
            self.bn3 = BatchNorm2d(32)
            w_next, h_next = self.cal_conv2d_image_size(w_next, h_next, kernel_size=1, stride=1)
            self.conv4 = Conv2d(32, 64, kernel_size=4, stride=2)
            self.bn4 = BatchNorm2d(64)
            w_next, h_next = self.cal_conv2d_image_size(w_next, h_next, kernel_size=4, stride=2)
        else:
            self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
            self.bn2 = BatchNorm2d(64)
            w_next, h_next = self.cal_conv2d_image_size(w_next, h_next, kernel_size=4, stride=2)
            #        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
            #        self.bn3 = BatchNorm2d(64)
            #        w_next, h_next = self.cal_conv2d_image_size(w_next, h_next, kernel_size=3, stride=1)
        
        lin_input_size = w_next * h_next * 64    
        
        # action layer
        self.a_lin1 = Linear(lin_input_size, out_features=128)
        self.a_relu = LeakyReLU()
        self.a_lin2 = Linear(128, self.n_outputs)
        
        # state value layer
        self.v_lin1 = Linear(lin_input_size, 128)
        self.v_relu = LeakyReLU()
        self.v_lin2 = Linear(128, 1)
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # loss -- somehow unstable; the simulation used manual calculation of MSE
        self.loss = nn.MSELoss()

        
    def cal_conv2d_image_size(self, w, h, kernel_size=3, stride=1):
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h
    
        
    def forward(self, x):
        if RESNET:
            x = F.relu(self.bn1(self.conv1(x))) 
            identity = x.clone() # [6, 32, 32]
            x = F.relu(self.bn2(self.conv2(x))) 
            x = F.relu(self.bn3(self.conv3(x) + identity)) 
            x = F.relu(self.bn4(self.conv4(x))) 
            
        else:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
    #        x = F.relu(self.bn3(self.conv3(x)))
    
        # print(x.size())
        x = x.view(x.size(0), -1) 
        # print(x.size())
        
        ax = self.a_relu(self.a_lin1(x))
        ax = self.a_lin2(ax)

        vx = self.v_relu(self.v_lin1(x))
        vx = self.v_lin2(vx)

        q = vx + (ax - ax.mean())
        return q
    
    
    # save a network
    def save(self, filename):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, filename)
    
    
    # load a netowrk
    def load(self, filename):
        try: 
            checkpoint = torch.load(filename)
            self.load_state_dict(checkpoint['model_state_dict']) # load a model
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except:
            print("loading error --- something wrong")
               

# class Agent
class Agent:
    def __init__(self, env, utils):
        self.env = env      
        self.network = DDQN(n_outputs=env.action_space.n).to(DEVICE)    
        self.target_network = DDQN(n_outputs=env.action_space.n).to(DEVICE) 
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        self.network_update_freq = NET_UPDATE_FREQ 
        self.network_sync_freq = NET_SYNC_FREQ
        self.replay_memory = deque(maxlen=MEMORY_CAPACITY)
        self.utils = utils
        
        self.gamma = GAMMA # discount factor
        self.rewards_list_last100eps = deque(maxlen=100) # reward list for the last 100 episodes
        self.reward_th = REWARD_TH
        self.steps = 0 # total steps
        self.episodes = 1
        self.epsilon = 1. # for epsilon-greedy; initially 1
        self.max_episodes = MAX_EPISODE        
        self.loss_list = [] # a list of the mean loss in each episode


    # preprocess an image
    def preprocess(self, image):
        image = image[20:,:] # remove score table on top
        img = Image.fromarray(image).convert('L') # RGB to Luminance
        img = img.resize(IMAGE_SIZE, Image.BILINEAR) # with bilinear filter, not critical 
        arr = np.array(img)/255.  # normalization; range [0, 1]
        img.close()
        return arr.T # np.array conversion switch the dimension (w,h)->(h,w)       
    
    
    def calculate_loss(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*random.sample(self.replay_memory, BATCH_SIZE))
        
        # batch to one array
        state_batch = np.concatenate(state_batch)
        next_state_batch = np.concatenate(next_state_batch)
                
        # convert them to tensors      
        state_t = torch.FloatTensor(state_batch).to(device=self.network.device)         
        next_state_t = torch.FloatTensor(next_state_batch).to(device=self.network.device)
        action_t = torch.LongTensor(action_batch).unsqueeze(1).to(device=self.network.device)
        reward_t = torch.FloatTensor(reward_batch).to(device=self.network.device)
        done_t = torch.ByteTensor(done_batch).to(device=self.network.device).float() 
        
        state_q = self.network(state_t)
        next_state_q = self.network(next_state_t)
        next_state_target_q = self.target_network(next_state_t)

        # Selected action's q_value
        q_pred = state_q.gather(1, action_t).squeeze(1)
        next_state_target_q = next_state_target_q.gather(1, next_state_q.max(1)[1].unsqueeze(1)).squeeze(1)
        q_target = reward_t + self.gamma * next_state_target_q * (1 - done_t) # done=1 -> zero
        
        #loss = self.network.loss(q_pred, q_target.detach()) # somehow MSE is not stable
        loss = (q_pred - q_target.detach()).pow(2).mean()
        max_q = torch.max(state_q).item()       
        return loss, max_q 

    
    def update_network(self):
        # update params of network
        if len(self.replay_memory)/MEMORY_CAPACITY < REPLAY_MEM_TH:  
            loss, max_q = [0, 0]
        else:
            # optim: zero_grad() -> loss -> backward() -> step()
            self.network.optimizer.zero_grad()
            loss, max_q = self.calculate_loss()
            loss.backward()
            self.network.optimizer.step()
            
        return loss, max_q
    
    
    def sync_network(self):
        # update params of target_network by sync it with network
        curr_state_dict = self.network.state_dict() # state_dict from the current network
        self.target_network.load_state_dict(curr_state_dict) # update target_network
        
        
    def train(self):
        while True: # loop for episodes
            curr_img = self.preprocess(self.env.reset()) # reset before each episode starts
            state = np.stack((curr_img, curr_img, curr_img, curr_img))
            rewards = 0
            episode_loss = 0
            episode_max_q = 0
            self.epsilon = self.adaptive_epsilon()
            done = False
            e_steps = 0
               
            while not done: # loop for steps
                done, state,  reward = self.take_step(state) 

                if self.steps % self.network_update_freq == 0:
                    loss, max_q = self.update_network() # calculate loss, update networks
                
                ###### sync networks at self.network_sync_freq  
                #if self.steps % self.network_sync_freq == 1:    
                #    self.sync_network() # sync networks
                
                self.steps += 1
                e_steps += 1
                
                rewards += reward
                episode_loss += loss
                episode_max_q += max_q/e_steps

            ###### sync networks every episode
            self.sync_network() # sync networks
            
            # update loss & reward once one episode is finished
            self.loss_list.append(episode_loss)
            self.rewards_list_last100eps.append(rewards)
        
            # monitor the progress
            str_out = 'episode: {}, tot_steps: {}, epi_steps: {}, loss: {:.2f}, ave_rew: {:.2f}, ave_max_q: {:.2f}, epsilon: {:.2f}' \
                        .format(self.episodes, self.steps, e_steps, self.loss_list[-1], \
                        np.mean(self.rewards_list_last100eps), episode_max_q, self.epsilon )
            print(str_out)
            
            if np.mean(self.rewards_list_last100eps) > self.reward_th or self.episodes > self.max_episodes:
                print(f'\nTraining done. Either max episodes reached or it passed the threshold.')
                self.save() # save the last model
                break 
                
            # checkpoint -- save
            utils.save_log(str_out) # save log
            if (self.episodes%SAVE_FREQ==0):
                self.save()
            
            self.episodes += 1 # start from episode 1
            gc.collect()

            
    def take_step(self, state):
        '''
        takes one time-step -- experience, update replay memory & the current state
        '''
        action = self.get_action(state)    
        next_img, reward, done, info = self.env.step(action)
        next_img = self.preprocess(next_img) # preprocess
        next_state = np.stack((next_img, state[0], state[1], state[2]))
        self.replay_memory.append((state[None,:], action, reward, next_state[None,:], done)) # 'None' creates a new axis
        state = next_state
        return done, state, reward            
    
    
    def get_action(self, state):
        '''
        if replay memory is not filled enough, random action
        '''
        # epsilon-greedy action
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample() # random action
        else:
            with torch.no_grad(): # not sure if not having this was the source of a bug
                state_t = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0) # local variable
                q = self.network(state_t)
                action = torch.argmax(q).item()
        return action

    
    def adaptive_epsilon(self):
        return max(self.epsilon*0.99, EPSILON_MIN) # epsilon decays from the beginning

    
    def save(self):
        '''
        save a trained network and an agent
        ''' 
        path = PATH
        Path(path).mkdir(parents=True, exist_ok=True)
        # network
        fname_net =  path + 'model_ep'+ str(self.episodes) +'.pt'
        self.network.save(fname_net)
        # agent
        fname_agnt =  path + 'agent_ep'+ str(self.episodes) +'.pkl'
        utils.save_object(agnt, fname_agnt)

        
# class Utils
class Utils:
    def __init__(self):
        self.path = PATH


    def save_object(self, obj, filename):
        '''
        save an object using pickle (e.g. Agent, Replay_Memory)
        '''
        fname = filename
        with open(fname, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        
    def load_object(self, filename):
        '''
        load an object (e.g. Agent, Replay_Memory) using piclke
        '''
        with open(filename, 'rb') as input:
            output = pickle.load(input)
            return output
     
        
    def save_log(self, str_out):
        filename = PATH + 'log_'+ ENVIRONMENT + '.txt'
        if os.path.exists(filename):
            file = open(filename, 'a') # append
        else:
            file = open(filename, 'w') # new file
        file.write(str_out + '\n')            
        file.close()


                


# ######### main ##########
if __name__ == '__main__':
    
    # initialization
    env = gym.make(ENVIRONMENT)
    utils = Utils()

    isRetraining = RETRAINING     
    if not isRetraining: # initial training
        agnt = Agent(env, utils)
    else: # retraining
        path_outputs = PATH
        fname_agent = path_outputs + 'agent_v11_ep1800.pkl'
        agnt = utils.load_object(fname_agent)
        
    # train
    agnt.train()
