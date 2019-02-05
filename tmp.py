from common.multiprocessing_env import SubprocVecEnv
import gym
import numpy as np
from sim import sim
import yaml
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

num_envs = 4


def make_env(i):

	with open("sim-config.yaml", "r") as handle:
		sim_config = yaml.load(handle)
	
	sim_config["use-seed"] = True
	sim_config["seed"] = i
	sim_config["render"] = False
	
	def _thunk():
		
		env = sim(sim_config)
		#print(sim_config)
		return env

	return _thunk

'''
def make_env(i):

	def _thunk():
		
		env = gym.make("CartPole-v0")
		#print(sim_config)
		return env

	return _thunk
'''
# Select Device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# Vector Envs
envs = [make_env(i) for i in range(num_envs)]
envs = SubprocVecEnv(envs)

# Test Env
with open("sim-config.yaml", "r") as handle:
	sim_config = yaml.load(handle)

env = sim(sim_config)
#env = gym.make("CartPole-v0")

# Network
class ActorCritic(nn.Module):

	def __init__(self, num_inputs, num_outputs, hidden_size):
		super(ActorCritic, self).__init__()

		self.critic = nn.Sequential(
			nn.Linear(num_inputs, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1))

		self.actor = nn.Sequential(
			nn.Linear(num_inputs, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, num_outputs),
			nn.Softmax(dim=1))

	def forward(self, x):

		value = self.critic(x)
		probs = self.actor(x)
		self.probs = probs
		dist = Categorical(probs)
		return dist, value


def test_env():

	state = env.reset()
	done = False

	total_reward = 0.0

	while not done:
		
		state = torch.FloatTensor(state).unsqueeze(0).to(device)
		dist, _ = model(state)
		next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
		state = next_state

		total_reward += reward

	return total_reward
		
def compute_returns(next_value, rewards, masks, gamma = 0.99):
	R = next_value
	returns = []

	for step in reversed(range(len(rewards))):
		R = rewards[step] + gamma * R * masks[step]
		returns.insert(0, R)

	return returns


num_inputs = envs.observation_space.shape[0]
num_outputs = envs.action_space.n
hidden_size = 128
lr = 3e-4
num_steps = 5
writer = SummaryWriter()

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)

max_frames = 2000000
frame_idx = 0
test_rewards = []

state = envs.reset()

while frame_idx < max_frames:

	log_probs = []
	values = []
	rewards = []
	masks = []
	entropy = 0

	for _ in range(num_steps):

		state = torch.FloatTensor(state).to(device)
		dist, value = model(state)

		action = dist.sample()
		
		next_state, reward, done, _ = envs.step(action.cpu().numpy())

		log_prob = dist.log_prob(action)
		entropy += dist.entropy().mean()

		log_probs.append(log_prob)
		values.append(value)
		rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
		masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))

		state = next_state
		frame_idx += 1

		if frame_idx % 10000 == 0:
			print(frame_idx)
			torch.save(model.state_dict(), "weights.torch")
			#test_rewards.append(np.mean([test_env() for _ in range(10)]))
			test_r_mean = np.mean([test_env() for _ in range(10)])
			writer.add_scalar("return graph", test_r_mean, frame_idx)

	next_state = torch.FloatTensor(next_state).to(device)
	_, next_value = model(next_state)
	returns = compute_returns(next_value, rewards, masks)

	log_probs = torch.cat(log_probs)
	returns = torch.cat(returns).detach()
	values = torch.cat(values)

	advantage = returns - values

	actor_loss = -(log_probs * advantage.detach()).mean()
	critic_loss = advantage.pow(2).mean()

	loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
