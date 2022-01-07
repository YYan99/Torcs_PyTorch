import os
import numpy as np
import math
import random
import datetime
import collections
import torch
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal 

from collections import deque
from torch.autograd import Variable as V
from gym_torcs import TorcsEnv
from tensorboardX import SummaryWriter

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

LOG_STD_MIN = 2
LOG_STD_MAX = -5

load_model = 1 # load model or not
train_indicator = 0 # train or not

state_size = 29
action_size = 3
BUFFER_SIZE = 100000
BATCH_SIZE = 32
policy_lr = 0.0001
q_lr = 0.001
ep_num = 2000
total_timesteps = 4000000
learning_starts = 5000
ts = 100000
EXPLORE = 100000
GAMMA = 0.95
TAU = 0.001
epsilon = 1
old_reward = 0
policy_delay = 2
policy_noise = 0.2
noise_clip = 0.2
policy_frequency = 1
target_network_frequency = 1



autotune = True
VISION = False

if (train_indicator):

    # model path
    path = './model/SAC/'+str(datetime.datetime.now())
    os.makedirs(path)

    # tensorboard
    writer = SummaryWriter('runs/SAC/'+str(datetime.datetime.now()), flush_secs=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('DEVICE:', device)
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False)

def layer_init(layer, weight_gain=1, bias_const=0):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
        nn.init.constant_(layer.bias, bias_const)

class Policy(nn.Module):
    def __init__(self, state_size, action_size, env):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(state_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        
        self.mean = nn.Linear(HIDDEN2_UNITS, action_size)
        self.logstd = nn.Linear(HIDDEN2_UNITS, action_size)

        self.action_scale = torch.FloatTensor((env.action_space.high - env.action_space.low) / 2)
        self.action_bias = torch.FloatTensor((env.action_space.high + env.action_space.low) / 2)
        self.apply(layer_init)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.logstd(x)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std
    
    def get_action(self, x, device):
        mean, log_std = self.forward(x, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t) # ??
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)

class SoftQNetwork(nn.Module):
    def __init__(self, state_size, action_size, layer_init):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size+action_size, HIDDEN1_UNITS)
        self.fc2 = nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS)
        self.fc3 = nn.Linear(HIDDEN2_UNITS, 1)
        self.apply(layer_init)

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), np.array(r_lst), np.array(s_prime_lst), np.array(done_mask_lst)

class OU():
    def function(self, x, mu, theta, sigma):
        # mu: mean 
        # theta: how fast the variable reverts towards to the mean
        # sigma: degree of volatility of the progress
        return theta * (mu - x) + sigma * np.random.randn(1) 

rb = ReplayBuffer(BUFFER_SIZE)
pg = Policy(state_size, action_size, env).to(device)
qf1 = SoftQNetwork(state_size, action_size, layer_init).to(device)
qf2 = SoftQNetwork(state_size, action_size, layer_init).to(device)
qf1_target = SoftQNetwork(state_size, action_size, layer_init).to(device)
qf2_target = SoftQNetwork(state_size, action_size, layer_init).to(device)
qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())
policy_optimizer = optim.Adam(list(pg.parameters()), policy_lr)
values_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), q_lr)
loss_fn = nn.MSELoss()

if autotune:
    target_entropy = - torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], q_lr)
else:
    alpha = 0.2

if load_model == 1:
    print("loading model")
    try:
        pg.load_state_dict(torch.load('./model/SAC/actormodel.pth'))
        pg.eval()
        qf1.load_state_dict(torch.load('./model/SAC/criticmodel.pth'))
        qf1.eval()
        qf2.load_state_dict(torch.load('./model/SAC/criticmodel1.pth'))
        qf2.eval()
        print("model load successfully")
    except:
        print("cannot find the model")

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

OU = OU()

global_episode = 0
episode_reward, episode_length = 0, 0

ob = env.reset()

for global_step in range(1, total_timesteps+1):

    done = False
    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 
    if (train_indicator):
        if global_step < learning_starts:
            action = env.action_space.sample()
        else:
            action, _, _ = pg.get_action([s_t], device)
    else:
        action, _, _ = pg.get_action([s_t], device)

        # a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        # if torch.cuda.is_available():
        #     a_t_original = a_t_original.data.cpu().numpy()
        # else:
        #     a_t_original = a_t_original.data.numpy()

        # # OU noise
        # noise_t[0][0] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][0], 0.0, 0.60, 0.30)
        # noise_t[0][1] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][1], 0.5, 1.00, 0.10)
        # noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], -0.1, 1.00, 0.05)

        # # stochastic brake
        # if random.random() <= 0.1:
        #     print("apply the brake")
        #     noise_t[0][2] = train_indicator * max(epsilon, 0) * OU.function(a_t_original[0][2], 0.2, 1.00, 0.10)

        # # action = original action + noise
        # a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
        # a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
        # a_t[0][2] = a_t_original[0][2] + noise_t[0][2]

        action = action.tolist()[0]

    ob, reward, done, _ = env.step(action)
    s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm)) 
    laptime = ob.lastLapTime
    rb.put((s_t, action, reward, s_t1, done))
    episode_reward += reward
    episode_length += 1
    if (train_indicator):
        if len(rb.buffer) > learning_starts:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(BATCH_SIZE)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = pg.get_action(s_next_obses, device)
                qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device)
                qf2_next_target = qf2_target.forward(s_next_obses, next_state_actions, device)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = torch.Tensor(s_rewards).to(device) + GAMMA * (1 - torch.Tensor(s_dones).to(device)) * (min_qf_next_target).view(-1)

            qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1) 
            qf2_a_values = qf2.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
            qf1_loss = loss_fn(qf1_a_values, next_q_value)
            qf2_loss = loss_fn(qf2_a_values, next_q_value)
            qf_loss = (qf1_loss + qf2_loss) / 2

            values_optimizer.zero_grad()
            qf_loss.backward()
            values_optimizer.step()

            if global_step % policy_frequency == 0:
                for _ in range(policy_frequency):
                    pi, log_pi, _ = pg.get_action(s_obs, device)
                    qf1_pi = qf1.forward(s_obs, pi, device)
                    qf2_pi = qf2.forward(s_obs, pi, device)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_optimizer.step()

                    if autotune:
                        with torch.no_grad():
                            _, log_pi, _ = pg.get_action(s_obs, device)
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                        
            if global_step % target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(TAU * param.data + (1-TAU) * target_param.data)

    if done:
        global_episode += 1
        
        print("---Episode ", global_episode ,  "  Reward:", episode_reward ," Laptime:", laptime)

        if (train_indicator):
            writer.add_scalar('Reward', episode_reward, global_step=global_episode)
            writer.add_scalar('Laptime', laptime, global_step=global_episode)

            # save best model
            if episode_reward > old_reward:
                print("saving model")
                torch.save(pg.state_dict(), path+'/actormodel_'+str(global_episode)+'.pth')
                torch.save(qf1.state_dict(), path+'/criticmodel_'+str(global_episode)+'.pth')
                torch.save(qf2.state_dict(), path+'/criticmodel1_'+str(global_episode)+'.pth')
                old_reward = episode_reward
            # save last model
            torch.save(pg.state_dict(), path+'/actormodel.pth')
            torch.save(qf1.state_dict(), path+'/criticmodel.pth')
            torch.save(qf2.state_dict(), path+'/criticmodel1.pth')

        episode_reward, episode_length = 0, 0
        if global_episode == ep_num:
            break
        if np.mod(global_episode, 3) == 0:
            ob = env.reset(relaunch=True) # necessary?
        else:
            ob = env.reset()
        
env.end()
print("Finish.")