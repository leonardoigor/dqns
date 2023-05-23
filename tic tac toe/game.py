import gym
from gym import spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    def __init__(self):
        self.grid_size = 3
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.action_space = spaces.Discrete(self.grid_size**2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(
            self.grid_size, self.grid_size), dtype=np.int32)

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        return self.grid

    def step(self, action):
        row, col = divmod(action, self.grid_size)
        if self.grid[row][col] != 0:
            return self.grid, -10, False, {}
        self.grid[row][col] = 1
        reward, done = self.get_reward_done()
        if not done:
            self.opponent_move()
            reward, done = self.get_reward_done()
        return self.grid, reward, done, {}

    def opponent_move(self):
        empties = np.argwhere(self.grid == 0)
        index = np.random.choice(empties.shape[0])
        self.grid[empties[index][0]][empties[index][1]] = -1

    def get_reward_done(self):
        for i in range(self.grid_size):
            if np.sum(self.grid[i, :]) == 3 or np.sum(self.grid[:, i]) == 3:
                return 1, True
            elif np.sum(self.grid[i, :]) == -3 or np.sum(self.grid[:, i]) == -3:
                return -1, True
        if np.trace(self.grid) == 3 or np.trace(np.fliplr(self.grid)) == 3:
            return 1, True
        elif np.trace(self.grid) == -3 or np.trace(np.fliplr(self.grid)) == -3:
            return -1, True
        elif np.count_nonzero(self.grid == 0) == 0:
            return 0, True
        else:
            return 0, False

    def render(self, mode='human'):
        print(self.grid)
