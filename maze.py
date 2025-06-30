import numpy as np
import random
import matplotlib.pyplot as plt
import time
import json


class Maze:
    """Generates a random maze using recursive backtracking algorithm."""
    def __init__(self, width=10, height=10, seed=None):
        self.width = width if width % 2 == 1 else width - 1
        self.height = height if height % 2 == 1 else height - 1
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.maze = np.ones((height, width), dtype=int)  # 1 = wall, 0 = path
        self.start = (0, 0)
        self.goal = (height - 1, width - 1)
        self._generate_maze()
        self.maze[self.start] = 0
        self.maze[self.goal] = 0

    def _generate_maze(self):
        """Recursive backtracking maze generator."""
        visited = np.zeros_like(self.maze)
        stack = []

        def neighbors(y, x):
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            random.shuffle(directions)
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width and visited[ny, nx] == 0:
                    yield (ny, nx, dy // 2, dx // 2)

        # Start from top-left corner
        y, x = 0, 0
        visited[y, x] = 1
        self.maze[y, x] = 0
        stack.append((y, x))

        while stack:
            y, x = stack[-1]
            found = False
            for ny, nx, wy, wx in neighbors(y, x):
                if visited[ny, nx] == 0:
                    visited[ny, nx] = 1
                    self.maze[y + wy, x + wx] = 0  # Remove wall
                    self.maze[ny, nx] = 0
                    stack.append((ny, nx))
                    found = True
                    break
            if not found:
                stack.pop()

    """def display(self):
        # Prints the maze to the console.
        display_map = {0: "  ", 1: "██"}
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (y, x) == self.start:
                    row += " S"
                elif (y, x) == self.goal:
                    row += " G"
                else:
                    row += display_map[self.maze[y, x]]
            print(row)"""

    def plot(maze):
        grid = maze.get_grid()
        fig, ax = plt.subplots()
        ax.imshow(grid, cmap='gray_r')
        ax.scatter(*maze.get_start()[::-1], c='green', s=100, label='Start')
        ax.scatter(*maze.get_goal()[::-1], c='red', s=100, label='Goal')
        ax.legend()
        plt.show()

    def get_grid(self):
        """Returns the maze as a NumPy array (0 = free, 1 = wall)."""
        return self.maze.copy()

    def get_start(self):
        return self.start

    def get_goal(self):
        return self.goal


class MazeManager:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.seeds = []

    def get_new_maze(self):
        seed = int(time.time() * 1000) % (2**32)
        self.seeds.append(seed)
        return Maze(width=self.width, height=self.height, seed=seed)

    def get_maze_by_index(self, i):
        seed = self.seeds[i]
        return Maze(width=self.width, height=self.height, seed=seed)

    def save_seeds(self, path="maze_seeds.json"):
        with open(path, "w") as f:
            json.dump(self.seeds, f)

    def load_seeds(self, path="maze_seeds.json"):
        with open(path, "r") as f:
            self.seeds = json.load(f)
