import torch
import random
import numpy as np
import torch.optim as optim
from collections import deque
from model import DQN
from env import RANEnvironment

class EdgeAgent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.env = RANEnvironment()

        self.policy_net = DQN()
        self.target_net = DQN()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=1000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.batch_size = 32

        self.logs = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.rand(2)
        with torch.no_grad():
            return self.policy_net(torch.FloatTensor(state)).numpy()

    def store(self, transition):
        self.memory.append(transition)

    def train_local(self, episodes=10):
        for _ in range(episodes):
            state = self.env.reset()
            action = self.select_action(state)

            next_state, reward, th, lat = self.env.step(action)
            self.store((state, action, reward, next_state))

            self.logs.append((reward, th, lat))
            self.learn()

        self.epsilon *= self.epsilon_decay

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        q_vals = self.policy_net(states)
        next_q = self.target_net(next_states).max(1)[0].detach()

        target = rewards + self.gamma * next_q
        loss = ((q_vals.mean(1) - target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_weights(self):
        return self.policy_net.state_dict()

    def set_weights(self, weights):
        self.policy_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
