import os
import numpy as np
import math
import random
import datetime
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from torch.autograd import Variable as V
from gym_torcs import TorcsEnv
from tensorboardX import SummaryWriter

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

load_model = 1 # load model or not
train_indicator = 0# train or not

state_size = 29
action_size = 3
BUFFER_SIZE = 100000
BATCH_SIZE = 32 #32
LRA = 0.0001
LRC = 0.001
ep_num = 2000
ts = 100000
EXPLORE = 100000
GAMMA = 0.95
TAU = 0.001
epsilon = 1
old_reward = 0

VISION = False

if (train_indicator):

    # model path
    path = './model/DDPG/'+str(datetime.datetime.now())
    os.makedirs(path)

    # tensorboard
    writer = SummaryWriter('runs/DDPG/'+str(datetime.datetime.now()), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        
        self.steering = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.steering.weight, 0, 1e-4) # torch.nn.init.normal_(tensor, mean=0, std=1) ~N(mean, std)
        self.acceleration = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.acceleration.weight, 0, 1e-4)
        self.brake = nn.Linear(HIDDEN2_UNITS, 1)
        nn.init.normal_(self.brake.weight, 0, 1e-4)

    def forward(self, x):
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        out1 = torch.tanh(self.steering(x))        
        out2 = torch.sigmoid(self.acceleration(x))
        out3 = torch.sigmoid(self.brake(x))
        out = torch.cat((out1, out2, out3), 1) # torch.cat((A,B,C), dim), dim = 0:row, dim = 1:list
        return out

class CriticNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.w1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.a1 = nn.Linear(action_size, HIDDEN2_UNITS)
        self.h1 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.h3 = nn.Linear(HIDDEN2_UNITS, HIDDEN2_UNITS)
        self.V = nn.Linear(HIDDEN2_UNITS, action_size)

    def forward(self, s, a):
        w1 = F.relu(self.w1(s))
        a1 = self.a1(a)
        h1 = self.h1(w1)
        h2 = h1 + a1
        h3 = F.relu(self.h3(h2))
        out = self.V(h3)
        return out

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):

        # every batch_size sample once
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
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

actor = ActorNetwork(state_size=state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size=state_size, action_size=action_size).to(device)

if load_model == 1:
    print("loading model")
    try:
        actor.load_state_dict(torch.load('./model/DDPG/actormodel.pth'))
        actor.eval()
        critic.load_state_dict(torch.load('./model/DDPG/criticmodel.pth'))
        critic.eval()
        print("model load successfully")
    except:
        print("cannot find the model")

buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum') # loss.sum()

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

OU = OU()

for i in range(ep_num):

    reward = 0
    Loss = 0 

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch=True) # necessary?
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

    for j in range(ts):

        # Init parameters
        loss = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])

        # original action
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()

        # noise
        noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        # stochastic brake
        if random.random() <= 0.1:
            print("apply the brake")
            noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)

        # action = original action + noise
        a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        ob, r_t, done, info = env.step(a_t[0]) # a_t[0]: steer, acc, brake

        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

        laptime = ob.lastLapTime
        # add to replay buffer
        buff.add(s_t, a_t[0], r_t, s_t1, done)

        batch = buff.getBatch(BATCH_SIZE)

        states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()
        actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
        rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
        new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
        dones = np.asarray([e[4] for e in batch])

        y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float() # same shape as actions

        # use target network to calculate target_q_value
        target_q_values = target_critic(new_states, target_actor(new_states)) # q(s_j+1, a^_j+1; omega^-), a^_j+1 = mu(s_j+1; theta^-)

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + GAMMA * target_q_values[k]

        if(train_indicator):

            # training
            # update critic network
            q_values = critic(states, actions)
            loss = criterion_critic(y_t, q_values)
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_critic.step() # update parameters

            a_for_grad = actor(states)
            a_for_grad.requires_grad_() # change require_grad False=>True, calculate gradiant automatically
            q_values_for_grad = critic(states, a_for_grad)
            critic.zero_grad()
            q_sum = q_values_for_grad.sum()
            q_sum.backward(retain_graph=True)
            grads = torch.autograd.grad(q_sum, a_for_grad) 
            # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False,only_inputs=True, allow_unused=False),
            # gradiant: -grads[0], actor+critic
            # input: a_for_grad, output: q_sum

            # update actor network
            act = actor(states)
            actor.zero_grad()
            act.backward(-grads[0])
            optimizer_actor.step()

            #print("soft updates target network")
            new_actor_state_dict = collections.OrderedDict()
            new_critic_state_dict = collections.OrderedDict()
            for var_name in target_actor.state_dict():
                new_actor_state_dict[var_name] = TAU * actor.state_dict()[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
            target_actor.load_state_dict(new_actor_state_dict)

            for var_name in target_critic.state_dict():
                new_critic_state_dict[var_name] = TAU * critic.state_dict()[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
            target_critic.load_state_dict(new_critic_state_dict)

        s_t = s_t1
        # print("---Episode:", i, ", Timesteps:", j,", Action:", a_t,", Reward:", r_t, ", Loss:", loss)
        reward = reward + r_t
        Loss = Loss + loss

        if done:
            break

    print("---Episode ", i ,  "  Reward:", reward, "  Loss:", Loss, "  Laptime:", laptime)

    if (train_indicator):
        writer.add_scalar('Reward', reward, global_step=i)
        writer.add_scalar('Loss', Loss, global_step=i)
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