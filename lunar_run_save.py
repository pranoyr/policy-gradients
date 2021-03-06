import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from matplotlib import animation
import matplotlib.pyplot as plt


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


env = gym.make('LunarLander-v2')
env.seed(args.seed)
torch.manual_seed(args.seed)


# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.shape[0]
# max_action = float(env.action_space.high[0])
# min_action = float(env.action_space.low[0])
# print(state_dim)
# print(action_dim)
# print(max_action)
# print(min_action)

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(8, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(p=0.6)
        self.affine3 = nn.Linear(128, 128)
        self.dropout3 = nn.Dropout(p=0.6)
        self.affine_last = nn.Linear(128, 4)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(x)
        x = F.relu(x)

        x = self.affine2(x)
        x = self.dropout2(x)
        x = F.relu(x)

        x = self.affine3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        action_scores = self.affine_last(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

checkpoint = torch.load('./models/lunar_lander.pth', map_location='cpu')
policy.load_state_dict(checkpoint['model_state_dict'])


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    # actions = torch.multinomial(probs,24)
    # print(actions)
    # policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


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
        frames = []
        for t in range(1, 100000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            frames.append(env.render(mode="rgb_array"))
            # print(state)
            if args.render:
                env.render()
            # policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        print(running_reward)
        save_frames_as_gif(frames)
        break
        # finish_episode()

        # state_dict = {'model_state_dict': policy.state_dict()}
        # torch.save(state_dict, './models/lunar_lander.pth')

        # if i_episode % args.log_interval == 0:
        #     print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
        #           i_episode, ep_reward, running_reward))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break


if __name__ == '__main__':
    main()
