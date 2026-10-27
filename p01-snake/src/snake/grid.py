from enum import IntEnum

import numpy as np

from snake.env import SnakeEnv


class Tile(IntEnum):
    EMPTY = 0
    BODY = 1
    HEAD = 2
    FOOD = 3


def to_grid(env: SnakeEnv) -> np.ndarray:
    grid = np.full((env.size, env.size), Tile.EMPTY, dtype=np.int8)
    rows, cols = zip(*env.snake, strict=True)
    grid[rows, cols] = Tile.BODY
    grid[env.head] = Tile.HEAD
    if env.food is not None:
        grid[env.food] = Tile.FOOD
    return grid
