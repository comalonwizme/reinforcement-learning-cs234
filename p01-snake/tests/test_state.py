from collections import deque

import numpy as np
import pytest

from snake.env import Action, Cell, Direction, SnakeEnv
from snake.state import FOOD_NONE, N_FOOD, N_STATES, danger, encode, food_position


def place(env: SnakeEnv, snake: list[Cell], direction: Direction, food: Cell | None) -> None:
    env.snake = deque(snake)
    env._occupied = set(snake)
    env.direction = direction
    env.food = food


def rotate(cell: Cell, size: int) -> Cell:
    r, c = cell
    return c, size - 1 - r


@pytest.fixture
def env() -> SnakeEnv:
    return SnakeEnv(np.random.default_rng(0))


def test_start_with_food_ahead(env: SnakeEnv) -> None:
    env.food = (7, 12)
    assert danger(env) == 0
    assert food_position(env) == 7
    assert encode(env) == 7


@pytest.mark.parametrize(
    ("food", "expected"),
    [
        ((7, 10), 7),
        ((5, 7), 3),
        ((9, 4), 2),
        ((7, 2), 1),
        ((2, 12), 6),
    ],
)
def test_food_position_relative_to_heading(env: SnakeEnv, food: Cell, expected: int) -> None:
    env.food = food
    assert food_position(env) == expected


def test_missing_food_has_its_own_code(env: SnakeEnv) -> None:
    env.food = None
    assert food_position(env) == FOOD_NONE
    assert encode(env) == FOOD_NONE


@pytest.mark.parametrize(
    ("snake", "direction", "expected"),
    [
        ([(7, 14), (7, 13), (7, 12)], Direction.RIGHT, 1 << Action.STRAIGHT),
        ([(0, 7), (0, 6), (0, 5)], Direction.RIGHT, 1 << Action.LEFT),
        ([(14, 7), (14, 6), (14, 5)], Direction.RIGHT, 1 << Action.RIGHT),
        (
            [(0, 14), (0, 13), (0, 12)],
            Direction.RIGHT,
            1 << Action.STRAIGHT | 1 << Action.LEFT,
        ),
        (
            [(7, 7), (8, 7), (8, 8), (7, 8), (6, 8)],
            Direction.UP,
            1 << Action.RIGHT,
        ),
    ],
)
def test_danger_bits_match_actions(
    env: SnakeEnv, snake: list[Cell], direction: Direction, expected: int
) -> None:
    place(env, snake, direction, food=(10, 0))
    assert danger(env) == expected


def test_encoding_is_rotation_invariant(env: SnakeEnv) -> None:
    snake = [(0, 3), (1, 3), (1, 4), (0, 4)]
    direction = Direction.UP
    food: Cell = (5, 10)
    codes = []
    for _ in range(4):
        place(env, snake, direction, food)
        codes.append(encode(env))
        snake = [rotate(cell, env.size) for cell in snake]
        food = rotate(food, env.size)
        direction = direction.turned(Action.RIGHT)
    expected = (1 << Action.STRAIGHT | 1 << Action.RIGHT) * N_FOOD + 2
    assert codes == [expected] * 4


def test_random_play_stays_in_range(env: SnakeEnv) -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            assert 0 <= encode(env) < N_STATES
            _, terminated, truncated = env.step(Action(rng.integers(len(Action))))
        assert 0 <= encode(env) < N_STATES
