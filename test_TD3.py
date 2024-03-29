import os
import numpy as np
import math
import random
import datetime
import collections
import tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from torch.autograd import Variable as V
from gym_torcs import TorcsEnv
from tensorboardX import SummaryWriter

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

load_model = 0 # load model or not
train_indicator = 1 # train or not
test_data = 0 # collect episode data

max_laptime = 300

state_size = 29 # 32 # 50 # 65 # 29
action_size = 3
BUFFER_SIZE = 100000
BATCH_SIZE = 32
LRA = 0.0001
LRC = 0.001
ep_num = 5000
ts = int(2e7) 
EXPLORE = 100000
GAMMA = 0.95 # 0.95
TAU = 0.001
epsilon = 1
old_reward = 0
old_laptime = 500
policy_delay = 2
policy_noise = 0.2
noise_clip = 0.2
VISION = False

past_t = 0

if (train_indicator):

    # model path
    path = './model/TD3/'+str(datetime.datetime.now())
    os.makedirs(path)

# tensorboard
if train_indicator or test_data:
    writer = SummaryWriter('runs/TD3/'+str(datetime.datetime.now()), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)

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
        out = torch.cat((out1, out2, out3), 1).squeeze(0) # torch.cat((A,B,C), dim), dim = 0:row, dim = 1:list
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

def test_agent():
        
    ob, d, ep_len, reward = env.reset(), 0, 0, 0
    o_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # array([angle, track, ...])

    while not (d or (ep_len >= max_laptime)):
        
        a_t = actor(torch.tensor(o_t.reshape(1, o_t.shape[0]), device=device).float())
        if torch.cuda.is_available():
            a_t = a_t.data.cpu().numpy()
        else:
            a_t = a_t.data.numpy()

        # Step
        ob, r, d, _ = env.step(a_t)
        o_t2 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

        reward += r
        ep_len += 1
        laptime = ob.lastLapTime

        o_t = o_t2

    return reward, ep_len, laptime

actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)
critic2 = CriticNetwork(state_size, action_size).to(device)

if load_model == 1:
    print("loading model")
    try:
        actor.load_state_dict(torch.load('./model/TD3/actormodel.pth'))
        actor.eval()
        critic.load_state_dict(torch.load('./model/TD3/criticmodel.pth'))
        critic.eval()
        critic2.load_state_dict(torch.load('./model/TD3/criticmodel2.pth'))
        critic2.eval()
        print("model load successfully")
    except:
        print("cannot find the model")

buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_critic2 = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()
target_critic2.load_state_dict(critic2.state_dict())
target_critic2.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum') # loss.sum()

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)
optimizer_critic2 = torch.optim.Adam(critic2.parameters(), lr=LRC)

env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

OU = OU()

for i in range(ep_num):

    reward = 0
    test_reward = 0
    test_laptime = 0
    test_timestep = 0
    cost_total = 0
    Loss = 0 
    Loss1 = 0 
    trackpos_avg = 0
    yaw_avg = 0
    a_max = 0

    if np.mod(i, 10) == 0:
        ob = env.reset(relaunch=True) # necessary?
    else:
        ob = env.reset()

    # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.aX, ob.aY, ob.aZ, ob.wheelSpinVel/100.0, ob.rpm)) # 1 + 19 + 1 + 3 + (65-29) + 4 + 1 = 29 + 3 = 32
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # 29
    # s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.aX, ob.aY, ob.aZ, ob.wheelSpinVel/100.0, ob.rpm, ob.distx, ob.disty, ob.speed_d)) # 1 + 19 + 1 + 3 + 3 + 4 + 1 + 6 + 6 + 6 = 50

    for j in range(ts):

        # Init parameters
        loss = 0
        loss1 = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([action_size])
        noise_t = np.zeros([action_size])

        # original action
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()

        if train_indicator:

            # OU noise
            noise_t[0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0], 0.0, 0.60, 0.30)
            noise_t[1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[1], 0.5, 1.00, 0.10)
            noise_t[2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[2], -0.1, 1.00, 0.05)

            # stochastic brake
            if random.random() <= 0.1:
                # print("apply the brake")
                noise_t[2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[2], 0.2, 1.00, 0.10)

        # action = original action + noise
        a_t[0] = a_t_original[0] + noise_t[0]
        a_t[1] = a_t_original[1] + noise_t[1]
        a_t[2] = a_t_original[2] + noise_t[2]

        # Step
        ob, r_t, done, cost = env.step(a_t) # a_t[0]: steer, acc, brake

        # s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.aX, ob.aY, ob.aZ, ob.wheelSpinVel/100.0, ob.rpm)) # 32
        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # 29
        # s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.aX, ob.aY, ob.aZ, ob.wheelSpinVel/100.0, ob.rpm, ob.distx, ob.disty, ob.speed_d))

        # Laptime Task reward
        if j + 1 >= max_laptime:
            done = 1
            if laptime > 0:
                r_t = 500
            
        # Senor data
        laptime = ob.lastLapTime
        trackpos = ob.trackPos
        yaw = ob.angle
        posx = ob.posX
        posy = ob.posY
        posz = ob.posZ
        ax = ob.aX
        ay = ob.aY
        az = ob.aZ
        distRaced = ob.distRaced
        racePos = ob.racePos
        trackpos_avg += np.abs(trackpos)  
        yaw_avg += np.abs(yaw)
        speed = ((300*ob.speedX)**2 + (300*ob.speedY)**2 + (300*ob.speedZ)**2)**0.5
        a = ((50*ob.aX)**2 + (50*ob.aY)**2 + (50*ob.aZ)**2)**0.5
        if a > a_max:
            a_max = a

        # Add data to tensorboard
        if test_data:
            writer.add_scalar('Trackpos/iteration_'+str(i), trackpos, global_step=j)
            writer.add_scalar('Angle/iteration_'+str(i), yaw, global_step=j)
            writer.add_scalar('Pos/X'+str(i), posx, global_step=j)
            writer.add_scalar('Pos/Y'+str(i), posy, global_step=j)
            writer.add_scalar('Pos/Z'+str(i), posz, global_step=j)
            writer.add_scalar('Speed/speed'+str(i), speed, global_step=j)
            writer.add_scalar('a/X'+str(i), ax, global_step=j)
            writer.add_scalar('a/Y'+str(i), ay, global_step=j)
            writer.add_scalar('a/Z'+str(i), az, global_step=j)
            writer.add_scalar('distRaced/distRaced_'+str(i), distRaced, global_step=j)
            writer.add_scalar('racePos/racePos_'+str(i), racePos, global_step=j)

        # Add to replay buffer
        buff.add(s_t, a_t, r_t, s_t1, done)

        # End of trajectory handling
        s_t = s_t1
        reward += r_t

        if done:
            trackpos_avg /= (j+1)
            yaw_avg /= (j+1)
            past_t += j + 1

            break

        # Update
        if train_indicator:

            batch = buff.getBatch(BATCH_SIZE)
            states = torch.tensor(np.asarray([e[0] for e in batch]), device=device).float()
            actions = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()
            rewards = torch.tensor(np.asarray([e[2] for e in batch]), device=device).float()
            new_states = torch.tensor(np.asarray([e[3] for e in batch]), device=device).float()
            dones = np.asarray([e[4] for e in batch])

            y_t = torch.tensor(np.asarray([e[1] for e in batch]), device=device).float()

            noise = torch.ones_like(actions).data.normal_(0, policy_noise).to(device=device) 
            noise = noise.clamp(-noise_clip, noise_clip)
            next_actions = (target_actor(new_states) + noise)
            
            # use target network to calculate target_q_value
            target_q_values = target_critic(new_states, next_actions) # q(s_j+1, a^_j+1; omega^-), a^_j+1 = mu(s_j+1; theta^-)
            target_q_values1 = target_critic2(new_states, next_actions) 
            
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA * torch.min(target_q_values[k], target_q_values1[k])

            # update critic network
            q_values = critic(states, actions)
            loss = criterion_critic(y_t, q_values)
            optimizer_critic.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_critic.step() # update parameters

            q_values2 = critic2(states, actions)
            loss2 = criterion_critic(y_t, q_values2)
            optimizer_critic2.zero_grad()
            loss2.backward(retain_graph=True)
            optimizer_critic2.step() # update parameters

            if j % policy_delay == 0:

                act = actor(states)
                actor_loss = -critic(states, act).mean()
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                new_actor_state_dict = collections.OrderedDict()
                new_critic_state_dict = collections.OrderedDict()
                new_critic2_state_dict = collections.OrderedDict()
                
                for var_name in target_actor.state_dict():
                    new_actor_state_dict[var_name] = TAU * actor.state_dict()[var_name] + (1-TAU) * target_actor.state_dict()[var_name]
                target_actor.load_state_dict(new_actor_state_dict)

                for var_name in target_critic.state_dict():
                    new_critic_state_dict[var_name] = TAU * critic.state_dict()[var_name] + (1-TAU) * target_critic.state_dict()[var_name]
                target_critic.load_state_dict(new_critic_state_dict)

                for var_name in target_critic2.state_dict():
                    new_critic2_state_dict[var_name] = TAU * critic2.state_dict()[var_name] + (1-TAU) * target_critic2.state_dict()[var_name]
                target_critic2.load_state_dict(new_critic2_state_dict)

        cost_total += cost
        Loss = Loss + loss
        Loss1 = Loss1 + loss1

    if train_indicator and (reward > 10000 or laptime > 0):
        test_reward, test_timestep, test_laptime = test_agent()

        # save best model
        if test_reward > old_reward:
            print("Saving Model")
            torch.save(actor.state_dict(), path+'/actormodel_'+str(i)+'_r='+str(test_reward)+'.pth')
            torch.save(critic.state_dict(), path+'/criticmodel_'+str(i)+'_r='+str(test_reward)+'.pth')
            torch.save(critic2.state_dict(), path+'/criticmodel2_'+str(i)+'_r='+str(test_reward)+'.pth')
            old_reward = test_reward
        if 0 < test_laptime < old_laptime:
            print("Saving Model")
            torch.save(actor.state_dict(), path+'/actormodel_'+str(i)+'_l='+str(test_laptime)+'.pth')
            torch.save(critic.state_dict(), path+'/criticmodel_'+str(i)+'_l='+str(test_laptime)+'.pth')
            torch.save(critic2.state_dict(), path+'/criticmodel2_'+str(i)+'_l='+str(test_laptime)+'.pth')
            old_laptime = test_laptime
        # save last model
        torch.save(actor.state_dict(), path+'/actormodel.pth')
        torch.save(critic.state_dict(), path+'/criticmodel.pth')
        torch.save(critic2.state_dict(), path+'/criticmodel2.pth')

    print('-------------------------------------',
            " \nEpisode: ".ljust(20),     i, 
            " \nTimestep:".ljust(20),   past_t, 
            " \nEpisode Length:".ljust(20),   j+1, 
            " \nReward: ".ljust(20),  "%.2f" %  reward, 
            " \nReward perstep:".ljust(20), "%.2f" % float(reward/(j+1)),
            " \nLaptime:".ljust(20),   laptime, 
            " \nTest Laptime:".ljust(20),   test_laptime, 
            " \nTest Reward:".ljust(20),  "%.2f" % test_reward, 
            " \nTest Timestep:".ljust(20),   test_timestep, 
           '\n-------------------------------------' )


    if train_indicator:
        writer.add_scalar('Reward', reward, global_step=i)
        writer.add_scalar('Reward_perstep', reward/j, global_step=i)
        writer.add_scalar('Cost', cost_total, global_step=i)
        writer.add_scalar('Loss/Q1', Loss, global_step=i)
        writer.add_scalar('Loss/Q2', Loss1, global_step=i)
        writer.add_scalar('Laptime', laptime, global_step=i)
        writer.add_scalar('AvgTrackpos', trackpos_avg, global_step=i)
        writer.add_scalar('AvgYaw', yaw_avg, global_step=i)
        writer.add_scalar('distRaced', distRaced, global_step=i)
        writer.add_scalar('a_max', a_max, global_step=i)
        writer.add_scalar('Test_Reward', test_reward, global_step=i)
        writer.add_scalar('Test_Laptime', test_laptime, global_step=i)
        writer.add_scalar('Test_Timestep', test_timestep, global_step=i)

env.end()
print("Finish.")