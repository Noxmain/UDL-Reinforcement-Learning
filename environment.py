import numpy as np

class MazeEnv:
    def __init__(self, maze):
        self.maze = maze.get_grid()
        self.start = maze.get_start()
        self.goal = maze.get_goal()
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        self.n_actions = len(self.actions)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action_idx):
        dy, dx = self.actions[action_idx]
        y, x = self.agent_pos
        ny, nx = y + dy, x + dx

        # Check bounds and walls
        if 0 <= ny < self.maze.shape[0] and 0 <= nx < self.maze.shape[1]:
            if self.maze[ny, nx] == 0:
                self.agent_pos = (ny, nx)

        reward = -1
        done = False
        if self.agent_pos == self.goal:
            reward = 10
            done = True

        return self.agent_pos, reward, done, {}

    def render(self):
        for y in range(self.maze.shape[0]):
            row = ""
            for x in range(self.maze.shape[1]):
                if (y, x) == self.agent_pos:
                    row += " A"
                elif (y, x) == self.start:
                    row += " S"
                elif (y, x) == self.goal:
                    row += " G"
                elif self.maze[y, x] == 1:
                    row += "██"
                else:
                    row += "  "
            print(row)
        print()
