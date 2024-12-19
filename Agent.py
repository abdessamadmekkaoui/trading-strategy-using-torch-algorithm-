import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval
        self.rewards = []

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.delta = 0.97
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if is_eval:
            self.model = self.load_model(model_name)
        else:
            self.model = DQN(state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def load_model(self, model_name):
        model = DQN(self.state_size, self.action_size).to(self.device)
        model.load_state_dict(torch.load(f"models/{model_name}"))
        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            options = self.model(state)
        return np.argmax(options.cpu().detach().numpy())

    def stockRewards(self, rewardto):
        self.rewards.append(rewardto)

    def expReplay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            state_tensor = torch.FloatTensor(state).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            
            # Get current Q values
            with torch.no_grad():
                current_q_values = self.model(state_tensor).cpu().detach().numpy()
            
            # Calculate target Q values
            if not done:
                with torch.no_grad():
                    next_q_values = self.model(next_state_tensor)
                    max_next_q = np.amax(next_q_values.cpu().detach().numpy())
                    target = reward + self.gamma * max_next_q
            else:
                target = reward
            
            # Update target Q value for the taken action
            target_f = current_q_values.copy()
            target_f[0][action] = target
            
            # Convert to tensor for training
            target_f_tensor = torch.FloatTensor(target_f).to(self.device)
            
            # Train the model
            self.optimizer.zero_grad()
            outputs = self.model(state_tensor)
            loss = nn.MSELoss()(outputs, target_f_tensor)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def getRewards(self):
        mini_batch = []
        rewards = []
        l = len(self.memory)

        for i in range(0, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            if reward > 0:
                rewards.append(reward)

        return rewards

    def getAgentsrewards(self):
        return self.rewards

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

if __name__ == "__main__":
    stock_name, window_size, episode_count = 'GOLD', 3, 10
    
    agent = Agent(window_size)
    data = getStockDataVec(stock_name)
    l = len(data) - 1
    batch_size = 32

    total_profitl = []
    buy_info = []
    sell_info = []
    data_Store = []

    for e in range(episode_count + 1):
        print("Episode " + str(e) + "/" + str(episode_count))
        state = getState(data, 0, window_size + 1)

        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1:  # buy
                agent.inventory.append(data[t])
                print("Buy: " + formatPrice(data[t]))
                buy_info.append(data[t])
                d = str(data[t]) + ', ' + 'Buy'
                data_Store.append(d)

            elif action == 2 and len(agent.inventory) > 0:  # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price

                print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))
                total_profitl.append(data[t] - bought_price)

                step_price = data[t] - bought_price
                info = str(data[t]) + ',' + str(step_price) + ',' + str(reward)
                sell_info.append(info)
                d = str(data[t]) + ', ' + 'Sell'
                data_Store.append(d)

            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print("--------------------------------")
                print("Total Profit: " + formatPrice(total_profit))
                print("--------------------------------")

            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)

        if e % 10 == 0:
            torch.save(agent.model.state_dict(), f"models/model_ep{e}")