import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
					help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
					help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
					help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
					help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('MountainCarContinuous-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)



state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])
print(state_dim)
print(action_dim)
print(max_action)
print(min_action)


class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.affine1 = nn.Linear(2, 128)
		self.dropout = nn.Dropout(p=0.6)
		self.affine2 = nn.Linear(128, 128)
		self.dropout2 = nn.Dropout(p=0.6)


		self.mu = nn.Linear(128, 1)
		# self.dropout3 = nn.Dropout(p=0.6)
		self.var = nn.Linear(128, 1)
		

		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		x = self.affine1(x)
		x = self.dropout(x)
		x = F.relu(x)

		x = self.affine2(x)
		x = self.dropout2(x)
		base = F.relu(x)

		mu = F.tanh(self.mu(base))
		# x = self.dropout3(x)

		var = F.softplus(self.mu(base))


		return mu.cpu().data.numpy(), var.cpu().data.numpy()



policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
	state = torch.from_numpy(state).float().unsqueeze(0)
	mu, var = policy(state)
	probs = np.random.normal(mu, var)


	actions = torch.from_numpy(np.clip(probs,-1 , 1))
	

	# probs = probs.normal_(mean=a[0].item(), std = a[1].item())
	m = Categorical(probs)
	action = m.sample()
	policy.saved_log_probs.append(m.log_prob(action))
	return action


def finish_episode():
	R = 0
	policy_loss = []
	returns = []
	for r in policy.rewards[::-1]:
		R = r + args.gamma * R
		returns.insert(0, R)
	returns = torch.tensor(returns)
	returns = (returns - returns.mean()) / (returns.std() + eps)
	for log_prob, R in zip(policy.saved_log_probs, returns):
		policy_loss.append(-log_prob * R)
	optimizer.zero_grad()
	policy_loss = torch.cat(policy_loss).sum()
	policy_loss.backward()
	optimizer.step()
	del policy.rewards[:]
	del policy.saved_log_probs[:]


def main():
	running_reward = 10
	for i_episode in count(1):
		state, ep_reward = env.reset(), 0
		for t in range(1, 100000):  # Don't infinite loop while learning
			print(state.shape)
			action = select_action(state)
			state, reward, done, _ = env.step(action)
			# print(state)
			# if args.render:
			env.render()
			policy.rewards.append(reward)
			ep_reward += reward
			# if state[0] > 0.5:
			# 	break
			if done:
			    print(t)
			    break

		running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
		print(running_reward)
		finish_episode()
	
		# if i_episode % args.log_interval == 0:
		#     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
		#           i_episode, ep_reward, running_reward))
		# if running_reward > env.spec.reward_threshold:
		#     print("Solved! Running reward is now {} and "
		#           "the last episode runs to {} time steps!".format(running_reward, t))
		#     break


if __name__ == '__main__':
	main()