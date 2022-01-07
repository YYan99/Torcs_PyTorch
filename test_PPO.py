import os
import numpy as np
import math
import random
import datetime
import collections
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque, namedtuple
from torch.autograd import Variable as V
from gym_torcs import TorcsEnv
from tensorboardX import SummaryWriter

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

load_model = 0 # load model or not
train_indicator = 1 # train or not

state_size = 29
action_size = 3
BUFFER_SIZE = 6000
BATCH_SIZE = 32 
LRA = 0.0002
LRC = 0.0002
ppo_epoch = 15 # 10
ep_num = 2000
ts = 100000
EXPLORE = 100000
# GAMMA = 0.9
# TAU = 0.001
epsilon = 1
old_reward = 0

clip_param = 0.2
max_grad_norm = 0.5
# critic_discount = 1e-6
entropy_beta = 0.007

VISION = False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_normal_density(x, mean, log_std, std):
    
    variance = std.pow(2)
    
    log_density = -(x-mean).pow(2) / (2*variance) - 0.5 * np.log(2*np.pi) - log_std
    #print(variance, log_std)
    log_density = log_density.sum(dim=1, keepdim=True)
    return log_density

if (train_indicator):

    # model path
    path = './model/PPO/'+str(datetime.datetime.now())
    os.makedirs(path)

    # tensorboard
    writer = SummaryWriter('runs/PPO/'+str(datetime.datetime.now()), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)

        self.fc3 = nn.Linear(HIDDEN2_UNITS, 3)
        self.logstd = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        x = torch.relu(self.fc1(x))        
        x = torch.relu(self.fc2(x))
        mu = self.fc3(x)
        logstd = self.logstd.expand_as(mu)
        sigma = torch.exp(logstd)
        action = torch.normal(mu, sigma)
        action = action.data.cpu().numpy()[0]
        action[0] = np.tanh(action[0])
        action[1] = sigmoid(action[1])
        action[2] = sigmoid(action[2])

        action = torch.as_tensor(action)
        action = action.view(1, -1)
        logprob = log_normal_density(action, mu, logstd, sigma)
        return action, logprob, mu

    def evaluate_actions(self, x, action):
        _, _, mean = self.forward(x)
        logstd = self.logstd.expand_as(mean)
        std = torch.exp(logstd)
        
        logprob = log_normal_density(action, mean, logstd, std)
        
        dist_entropy = 0.5 + 0.5 * math.log(2*math.pi) * logstd
        dist_entropy = dist_entropy.sum(-1).mean()
        return logprob, dist_entropy

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS, 1)

    def forward(self, x):
        v = torch.relu(self.fc1(x))
        v = torch.relu(self.fc2(v))
        value = self.fc3(v)
        return value

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = []

    def getBuffer(self):

        return self.buffer
    
    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, action_log_prob, done, value):
        experience = (state, action, reward, new_state, action_log_prob, done, value)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft() # list.pop(0) : remove first element
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class OU():
    def function(self, x, mu, theta, sigma):
        # mu: mean 
        # theta: how fast the variable reverts towards to the mean
        # sigma: degree of volatility of the progress
        return theta * (mu - x) + sigma * np.random.randn(1) 

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size).to(device)

if load_model == 1:
    print("loading model")
    try:
        actor.load_state_dict(torch.load('./model/PPO/actormodel.pth'))
        actor.eval()
        critic.load_state_dict(torch.load('./model/PPO/criticmodel.pth'))
        critic.eval()
        print("model load successfully")
    except:
        print("cannot find the model")

buff = ReplayBuffer(BUFFER_SIZE)

optimizer_actor = optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = optim.Adam(critic.parameters(), lr=LRC)

env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

OU = OU()

buffer = []

for i in range(ep_num):

    reward = 0
    sum_Loss = 0 
    sum_a_loss = 0
    sum_c_loss = 0
    Loss = 0 
    a_loss = 0
    c_loss = 0


    if np.mod(i, 10) == 0:
        ob = env.reset(relaunch=True) # necessary?
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

    for j in range(ts):
        
        epsilon -= 1/EXPLORE

        if (train_indicator):
            alpha = 0 # max(epsilon, 0)
        else:
            alpha = 0

        action, logprob, mu = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())
        action, logprob, mu = action.data.cpu().numpy()[0], logprob.data.cpu().numpy()[0], mu.data.cpu().numpy()[0]

        value = critic(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())
        value = value.data.cpu().numpy()[0]

        action[0] += alpha * OU.function(action[0], 0.0, 0.60, 0.30) 
        action[1] += alpha * OU.function(action[1], 0.5, 1.00, 0.10) 
        action[2] += alpha * OU.function(action[2], -0.1, 1.00, 0.05) 

        # stochastic brake
        if random.random() <= 0.1:
            print("apply the brake")
            action[2] += alpha * OU.function(action[2], 0.2, 1.00, 0.10)        
        print('Action:', action)
        next_ob, r_t, done, info = env.step(action)
        s_t1 = np.hstack((next_ob.angle, next_ob.track, next_ob.trackPos, next_ob.speedX, next_ob.speedY, next_ob.speedZ, next_ob.wheelSpinVel/100.0, next_ob.rpm)) 
        laptime = next_ob.lastLapTime

        if (train_indicator):

            buff.add(s_t, action, r_t, s_t1, logprob, done, value[0])
            buffer = buff.getBuffer()

            if buff.count() % BUFFER_SIZE == 0 :

                states = torch.tensor(np.asarray([e[0] for e in buffer]), device=device).float()
                actions = torch.tensor(np.asarray([e[1] for e in buffer]), device=device).float()
                rewards = torch.tensor(np.asarray([e[2] for e in buffer]), device=device).float().unsqueeze(1)
                new_states = torch.tensor(np.asarray([e[3] for e in buffer]), device=device).float()
                action_log_probs = torch.tensor(np.asarray([e[4] for e in buffer]), device=device).float()
                dones = np.asarray([e[5] for e in buffer])
                values = torch.tensor(np.asarray([e[6] for e in buffer]), device=device).float().unsqueeze(1)
            
                advantages = rewards - values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

                Loss = 0 
                a_loss = 0
                c_loss = 0

                for _ in range(ppo_epoch):
                    for index in BatchSampler(SubsetRandomSampler(range(BUFFER_SIZE)), BATCH_SIZE, False):
                        new_logprob, dist_entropy = actor.evaluate_actions(states[index], actions[index])
                        new_value = critic(states[index])
                        action_log_probs[index] = action_log_probs[index].view(-1, 1)
                        ratio = torch.exp(new_logprob - action_log_probs[index])
                        # print(advantages.size())
                        advantages[index] = advantages[index].view(-1, 1)
                        surrogate1 = ratio * advantages[index]
                        surrogate2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages[index]
                        policy_loss = - torch.min(surrogate1, surrogate2).mean()

                        rewards[index] = rewards[index].view(-1, 1)
                        value_loss = F.mse_loss(new_value, rewards[index])

                        loss = policy_loss + value_loss - entropy_beta * dist_entropy
                        print('LOSS:', loss, policy_loss, value_loss)
                        optimizer_actor.zero_grad()
                        optimizer_critic.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                        optimizer_actor.step()
                        optimizer_critic.step()

                buff.erase()

        s_t = s_t1
        reward += r_t

        if done:
            break

    print("---Episode ", i ,  "  Reward:", reward, "  Laptime:", laptime) #,  "  Loss:", sum_Loss)
    if (train_indicator):
        writer.add_scalar('Reward', reward, global_step=i)
        writer.add_scalar('Laptime', laptime, global_step=i)
        # save best model
        if reward > old_reward:
            print("saving model")
            torch.save(actor.state_dict(), path+'/actormodel_'+str(i)+'.pth')
            torch.save(critic.state_dict(), path+'/criticmodel_'+str(i)+'.pth')
            
            old_reward = reward
        # save last model
        torch.save(actor.state_dict(), path+'/actormodel.pth')
        torch.save(critic.state_dict(), path+'/criticmodel.pth')
    
            
env.end()
print("Finish.")