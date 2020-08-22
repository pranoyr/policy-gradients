import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torchvision.models import resnet18


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


# env = gym.make('SuperMarioBros-1-1-v0')
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.layer = resnet18(pretrained=False)
		self.layer.fc = nn.Sequential(
			nn.Linear(512, 5))
			
		self.saved_log_probs = []
		self.rewards = []

	def forward(self, x):
		x = self.layer(x)
		return F.softmax(x, dim=1)


# class Policy(nn.Module):
# 	def __init__(self):
# 		super(Policy, self).__init__()
# 		self.affine1 = nn.Linear(6, 128)
# 		self.dropout = nn.Dropout(p=0.6)
# 		self.affine2 = nn.Linear(128, 128)
# 		self.dropout2 = nn.Dropout(p=0.6)
# 		self.affine3 = nn.Linear(128, 128)
# 		self.dropout3 = nn.Dropout(p=0.6)
# 		self.affine_last = nn.Linear(128, 3)


# 		self.saved_log_probs = []
# 		self.rewards = []

# 	def forward(self, x):
# 		x = self.affine1(x)
# 		x = self.dropout(x)
# 		x = F.relu(x)

# 		x = self.affine2(x)
# 		x = self.dropout2(x)
# 		x = F.relu(x)

# 		x = self.affine3(x)
# 		x = self.dropout3(x)
# 		x = F.relu(x)
# 		action_scores = self.affine_last(x)
# 		return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
	state = state.copy()
	state = torch.from_numpy(state).float().unsqueeze(0).permute(0, 3, 1, 2)
	probs = policy(state)
	m = Categorical(probs)
	action = m.sample()
	policy.saved_log_probs.append(m.log_prob(action))
	return action.item()


def finish_episode():
	R = 0
	policy_loss = []
	returns = []
	print(policy.rewards)
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
		for t in range(1, 500):  # Don't infinite loop while learning
			action = select_action(state)
			state, reward, done, _ = env.step(action)
			# print(state)
			# if args.render:
			env.render()
			policy.rewards.append(reward)
			ep_reward += reward
			print(ep_reward)
			if done:
				print(f"finised episode at iteration {t}")
				break

		running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
		print(f"Running Reward : {running_reward}")
		finish_episode()

		state = {'model_state_dict': policy.state_dict()}
		torch.save(state, './models/model.pth')

		# if i_episode % args.log_interval == 0:
		#     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
		#           i_episode, ep_reward, running_reward))
		# if running_reward > env.spec.reward_threshold:
		#     print("Solved! Running reward is now {} and "
		#           "the last episode runs to {} time steps!".format(running_reward, t))
		#     break


if __name__ == '__main__':
	main()
