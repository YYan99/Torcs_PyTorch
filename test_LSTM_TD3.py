from copy import deepcopy
import os
from turtle import shape
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

########### LSTM_TD3 #########

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

load_model = 0 # load model or not
train_indicator = 1 # train or not
test_data = 0 # collect episode data

TASK = 1 # '1': laptime, '0': lanekeep
max_laptime = 300 # track3: 300 track2: 200 track1: 150



state_size = 29 # 32 # 29
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

max_hist_len = 200 # 100
past_t = 0

if (train_indicator):

    # model path
    path = './model/LSTM_TD3/'+str(datetime.datetime.now())
    os.makedirs(path)

# tensorboard
if train_indicator or test_data:
    writer = SummaryWriter('runs/LSTM_TD3/'+str(datetime.datetime.now()), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)

class ActorNetwork(nn.Module):
    def __init__(self, state_size):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)

        self.layer1 = nn.Linear(state_size+action_size, 128)
        self.layer2 = nn.LSTM(128, 128, batch_first=True)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()

        self.steering = nn.Linear(256, 1)
        nn.init.normal_(self.steering.weight, 0, 1e-4) # torch.nn.init.normal_(tensor, mean=0, std=1) ~N(mean, std)
        self.acceleration = nn.Linear(256, 1)
        nn.init.normal_(self.acceleration.weight, 0, 1e-4)
        self.brake = nn.Linear(256, 1)
        nn.init.normal_(self.brake.weight, 0, 1e-4)

    def forward(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):

        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None): 
            hist_obs = torch.zeros(1, 1, state_size).to(device)
            hist_act = torch.zeros(1, 1, action_size).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        x = torch.cat([hist_obs, hist_act], dim=-1)

        # pre-LSTM
        x = self.layer1(x)
        x = self.relu(x)
        # LSTM
        x, (lstm_hidden_state, lstm_cell_state) = self.layer2(x)
        x = self.relu(x)
        # after-LSTM
        x = self.layer3(x)
        x = self.relu(x)
        hist_out = torch.gather(x, 1, (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()).squeeze(1)

        x = obs
        x = self.layer4(x)
        x = self.relu(x)

        x = torch.cat([hist_out, x], dim=-1) # torch.Size([1, 128]) torch.Size([1, 128]) -> torch.Size([1, 256])
        
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

        self.layer1 = nn.Linear(state_size+action_size, 128)
        self.layer2 = nn.LSTM(128, 128, batch_first=True)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(state_size+action_size, 128)
        self.relu = nn.ReLU()

        self.layer5 = nn.Linear(256, 1)

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):

        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        x = torch.cat([hist_obs, hist_act], dim=-1)

        # pre-LSTM
        x = self.layer1(x)
        x = self.relu(x)
        # LSTM
        x, (lstm_hidden_state, lstm_cell_state) = self.layer2(x)
        x = self.relu(x)
        # after-LSTM
        x = self.layer3(x)
        x = self.relu(x)
        hist_out = torch.gather(x, 1, (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, 128).unsqueeze(1).long()).squeeze(1)

        x = torch.cat([obs, act], dim=-1)
        x = self.layer4(x)
        x = self.relu(x)

        x = torch.cat([hist_out,x], dim=-1)

        out = self.layer5(x)

        return out

class ReplayBuffer():
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, state_size), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, state_size), dtype=np.float32)
        self.act_buf = np.zeros((max_size, action_size), dtype=np.float32)
        self.rew_buf = np.zeros(max_size,dtype=np.float32)
        self.done_buf = np.zeros(max_size,dtype=np.float32)
        self.ptr, self.size = 0, 0 # num_experiences, buffer_size
        self.buffer = deque()

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size) # (low, high, size)
        
        # History
        hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
        hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
        hist_obs_len = max_hist_len * np.ones(batch_size)
        hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
        hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
        hist_obs2_len = max_hist_len * np.ones(batch_size)

        # Extract history experiences before sampled index
        for i, id in enumerate(idxs): # enumerate(): array([5,8,10]) -> 0 5, 1 8, 2 10
            hist_start_id = id - max_hist_len
            if hist_start_id < 0:
                hist_start_id = 0
            # If exist done before the last experience (not including the done in id), start from the index next to the done.
            if len(np.where(self.done_buf[hist_start_id:id] == 1)[0]) != 0:
                hist_start_id += np.where(self.done_buf[hist_start_id:id] == 1)[0][-1] + 1
            hist_seg_len = id - hist_start_id
            hist_obs_len[i] = hist_seg_len
            # print(hist_obs.shape, i, hist_seg_len, self.obs_buf.shape, hist_start_id, id)
            hist_obs[i, :hist_seg_len, :] = self.obs_buf[hist_start_id:id]
            hist_act[i, :hist_seg_len, :] = self.act_buf[hist_start_id:id]
            # If the first experience of an episode is sampled, the hist lengths are different for obs and obs2.
            if hist_seg_len == 0:
                hist_obs2_len[i] = 1
            else:
                hist_obs2_len[i] = hist_seg_len
            hist_obs2[i, :hist_seg_len, :] = self.obs2_buf[hist_start_id:id]
            hist_act2[i, :hist_seg_len, :] = self.act_buf[hist_start_id+1:id+1]

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)

# Set up function for computing TD3 Q-losses
def compute_loss_q(data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
    h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

    q1 = critic(o, a, h_o, h_a, h_o_len)
    q2 = critic2(o, a, h_o, h_a, h_o_len)

    # Bellman backup for Q functions
    with torch.no_grad():
        pi_targ = target_actor(o2, h_o2, h_a2, h_o2_len)

        # Target policy smoothing
        epsilon = torch.randn_like(pi_targ) * policy_noise
        epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
        a2 = pi_targ + epsilon
        a2[0] = torch.clamp(a2[0], -1, 1)
        a2[1] = torch.clamp(a2[1], 0, 1)
        a2[2] = torch.clamp(a2[2], 0, 1)
        
        # Target Q-values
        q1_pi_targ = target_critic(o2, a2, h_o2, h_a2, h_o2_len)
        q2_pi_targ = target_critic2(o2, a2, h_o2, h_a2, h_o2_len)       
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        
        backup = r + GAMMA * (1 - d) * q_pi_targ

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup) ** 2).mean()
    loss_q2 = ((q2 - backup) ** 2).mean()

    loss_q = loss_q1 + loss_q2

    return loss_q

# Set up function for computing TD3 pi loss
def compute_loss_pi(data):
    o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
    a = actor(o, h_o, h_a, h_o_len)
    q1_pi = critic(o, a, h_o, h_a, h_o_len)
    
    return -q1_pi.mean()

def test_agent():
        
    ob, d, ep_len, reward = env.reset(), 0, 0, 0
    o_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # array([angle, track, ...])


    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, state_size])
        a_buff = np.zeros([max_hist_len, action_size])
        o_buff[0, :] = o_t
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, state_size])
        a_buff = np.zeros([1, action_size])
        o_buff_len = 0

    while not d: # or (ep_len >= 300)):

        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)

        
        a_t = actor(torch.tensor(o_t.reshape(1, o_t.shape[0]), device=device).float(), h_o, h_a, h_l)
        if torch.cuda.is_available():
            a_t = a_t.data.cpu().numpy()
        else:
            a_t = a_t.data.numpy()

        # Step
        ob, r, d, _ = env.step(a_t)
        o_t2 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

        if TASK and ep_len >= max_laptime:
            d = 1

        reward += r
        ep_len += 1
        laptime = ob.lastLapTime

        # Add short history
        if max_hist_len != 0:
            if o_buff_len == max_hist_len:
                o_buff[:max_hist_len - 1] = o_buff[1:]
                a_buff[:max_hist_len - 1] = a_buff[1:]
                o_buff[max_hist_len - 1] = list(o_t)
                a_buff[max_hist_len - 1] = list(a_t)
            else:
                o_buff[o_buff_len + 1 - 1] = list(o_t)
                a_buff[o_buff_len + 1 - 1] = list(a_t)
                o_buff_len += 1
        o_t = o_t2

    return reward, ep_len, laptime

actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)
critic2 = CriticNetwork(state_size, action_size).to(device)

if load_model == 1:
    print("loading model")
    try:
        actor.load_state_dict(torch.load('./model/LSTM_TD3/actormodel.pth'))
        actor.eval()
        critic.load_state_dict(torch.load('./model/LSTM_TD3/criticmodel.pth'))
        critic.eval()
        critic2.load_state_dict(torch.load('./model/LSTM_TD3/criticmodel2.pth'))
        critic2.eval()
        print("model load successfully")
    except:
        print("cannot find the model")

buff = ReplayBuffer(obs_dim=state_size,act_dim=action_size,max_size=BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_critic2 = CriticNetwork(state_size, action_size).to(device)

target_actor.load_state_dict(actor.state_dict())
# target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
# target_critic.eval()
target_critic2.load_state_dict(critic2.state_dict())
# target_critic1.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum') # loss.sum()

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)
optimizer_critic2 = torch.optim.Adam(critic2.parameters(), lr=LRC)


env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

for i in range(ep_num):

    reward = 0
    test_reward = 0
    test_laptime = 0
    test_timestep = 0
    cost_total = 0
    Loss = 0 
    Loss1 = 0 
    Loss2 = 0 
    trackpos_avg = 0
    yaw_avg = 0
    a_max = 0

    if np.mod(i, 10) == 0:
        ob = env.reset(relaunch=True) # necessary?
    else:
        ob = env.reset()

    # o_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.aX, ob.aY, ob.aZ, ob.wheelSpinVel/100.0, ob.rpm)) 
    o_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) # array([angle, track, ...])

    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, state_size]) # array([[0,0,...], ...])
        a_buff = np.zeros([max_hist_len, action_size])
        o_buff[0, :] = o_t
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, state_size])
        a_buff = np.zeros([1, action_size])
        o_buff_len = 0

    # Main Loop
    for j in range(ts):

        # Init parameters
        loss = 0
        loss1 = 0
        loss2 = 0
        a_t = np.zeros([action_size])

        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)

        if past_t + j > 1000 or not train_indicator:
            # Original action
            a_t = actor(torch.tensor(o_t.reshape(1, o_t.shape[0]), device=device).float(), h_o, h_a, h_l)
            if torch.cuda.is_available():
                a_t = a_t.data.cpu().numpy()
            else:
                a_t = a_t.data.numpy()
        else:
            a_t = env.action_space.sample()

        # Step
        ob, r_t, done, cost = env.step(a_t) # a_t: steer, acc, brake

        # o_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.aX, ob.aY, ob.aZ, ob.wheelSpinVel/100.0, ob.rpm)) 
        o_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 

        # Laptime Task reward
        if TASK and j + 1 >= max_laptime:
            done = 1
            r_t = 500
            # if laptime > 0:
                

        # Sensor data
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
        # print('racePos',racePos)
        trackpos_avg += np.abs(trackpos)  
        yaw_avg += np.abs(yaw)
        speed = ((300*ob.speedX)**2 + (300*ob.speedY)**2 + (300*ob.speedZ)**2)**0.5
        acc = ((50*ob.aX)**2 + (50*ob.aY)**2 + (50*ob.aZ)**2)**0.5
        if acc > a_max:
            a_max = acc
        
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
        buff.store(o_t, a_t, r_t, o_t1, done)

        # Add short history
        if max_hist_len != 0:
            if o_buff_len == max_hist_len:
                # Remove first data
                o_buff[:max_hist_len-1] = o_buff[1:]
                a_buff[:max_hist_len-1] = a_buff[1:]
                # Add last data
                o_buff[max_hist_len-1] = list(o_t)
                a_buff[max_hist_len-1] = list(a_t)
            else:
                o_buff[o_buff_len] = list(o_t)
                a_buff[o_buff_len] = list(a_t)
                o_buff_len += 1

        o_t = o_t1
        reward += r_t

        # End of trajectory handling
        if done:  # or (j >= 200-1):            
            trackpos_avg /= (j+1)
            yaw_avg /= (j+1)
            past_t += j + 1

            break
         
        # Update 
        if train_indicator and past_t + j > 1000:

            batch = buff.sample_batch_with_history(batch_size=BATCH_SIZE, max_hist_len=max_hist_len)
            data = {k: v.to(device) for k, v in batch.items()}

            optimizer_critic.zero_grad()
            optimizer_critic2.zero_grad()
            loss_q = compute_loss_q(data)
            loss_q.backward()
            optimizer_critic.step()
            optimizer_critic2.step()

            if j % policy_delay == 0:

                # q_params.requires_gard = False
                for p in critic.parameters():
                    p.requires_gard = False
                for p in critic2.parameters():
                    p.requires_gard = False

                optimizer_actor.zero_grad()
                loss_pi = compute_loss_pi(data)
                loss_pi.backward()
                optimizer_actor.step()

                # q_params.requires_gard = True
                for p in critic.parameters():
                    p.requires_gard = True
                for p in critic2.parameters():
                    p.requires_gard = True

                with torch.no_grad():
                    for p, p_targ in zip(actor.parameters(), target_actor.parameters()):
                        p_targ.data.mul_(TAU)
                        p_targ.data.add_((1 - TAU) * p.data)
                    for p, p_targ in zip(critic.parameters(), target_critic.parameters()):
                        p_targ.data.mul_(TAU)
                        p_targ.data.add_((1 - TAU) * p.data)
                    for p, p_targ in zip(critic2.parameters(), target_critic2.parameters()):
                        p_targ.data.mul_(TAU)
                        p_targ.data.add_((1 - TAU) * p.data)

        Loss += loss
        Loss1 += loss1
        Loss2 += loss2

    if train_indicator and past_t + j > 1000 and (reward > 10000 or laptime > 0):
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
        writer.add_scalar('Reward_perstep', reward/(j+1), global_step=i)
        writer.add_scalar('Cost', cost_total, global_step=i)
        writer.add_scalar('Loss/Q1', Loss, global_step=i)
        writer.add_scalar('Loss/Q2', Loss1, global_step=i)
        writer.add_scalar('Loss/Q3', Loss2, global_step=i)
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