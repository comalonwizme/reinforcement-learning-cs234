import numpy as np

from snake.env import SnakeEnv
from snake.grid import Tile, to_grid


def test_grid_marks_head_body_and_food() -> None:
    env = SnakeEnv(np.random.default_rng(0))
    env.food = (0, 0)
    grid = to_grid(env)
    assert grid.shape == (env.size, env.size)
    assert grid[7, 7] == Tile.HEAD
    assert grid[7, 6] == Tile.BODY
    assert grid[7, 5] == Tile.BODY
    assert grid[0, 0] == Tile.FOOD
    assert np.count_nonzero(grid == Tile.EMPTY) == env.size**2 - 4


def test_grid_without_food() -> None:
    env = SnakeEnv(np.random.default_rng(0))
    env.food = None
    assert np.count_nonzero(to_grid(env) == Tile.FOOD) == 0
