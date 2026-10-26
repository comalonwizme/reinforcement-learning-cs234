from collections import deque
from itertools import pairwise

import numpy as np
import pytest

from snake.env import (
    INITIAL_LENGTH,
    REWARD_DEATH,
    REWARD_FOOD,
    REWARD_STEP,
    Action,
    Cell,
    Direction,
    SnakeEnv,
)

FAR_AWAY: Cell = (0, 0)
ALIVE = (REWARD_STEP, False, False)


def make_env(**kwargs: int) -> SnakeEnv:
    env = SnakeEnv(np.random.default_rng(0), **kwargs)
    env.food = FAR_AWAY
    return env


def eat_ahead(env: SnakeEnv) -> None:
    dr, dc = env.direction.delta
    r, c = env.head
    env.food = (r + dr, c + dc)
    assert env.step(Action.STRAIGHT) == (REWARD_FOOD, False, False)
    env.food = FAR_AWAY


def assert_consistent(env: SnakeEnv) -> None:
    body = list(env.snake)
    board = {(r, c) for r in range(env.size) for c in range(env.size)}
    assert len(set(body)) == len(body)
    assert set(body) <= board
    assert {cell for cell in board if env.is_blocked(cell)} == set(body)
    assert all(abs(r1 - r2) + abs(c1 - c2) == 1 for (r1, c1), (r2, c2) in pairwise(body))
    assert env.food is None or not env.is_blocked(env.food)


def food_sequence(seed: int, n_resets: int) -> list[Cell | None]:
    env = SnakeEnv(np.random.default_rng(seed))
    foods = [env.food]
    for _ in range(n_resets):
        env.reset()
        foods.append(env.food)
    return foods


@pytest.mark.parametrize(
    ("direction", "action", "expected"),
    [
        (Direction.UP, Action.STRAIGHT, Direction.UP),
        (Direction.UP, Action.LEFT, Direction.LEFT),
        (Direction.UP, Action.RIGHT, Direction.RIGHT),
        (Direction.RIGHT, Action.LEFT, Direction.UP),
        (Direction.LEFT, Action.LEFT, Direction.DOWN),
        (Direction.LEFT, Action.RIGHT, Direction.UP),
    ],
)
def test_turned(direction: Direction, action: Action, expected: Direction) -> None:
    assert direction.turned(action) is expected


@pytest.mark.parametrize("kwargs", [{"size": 3}, {"chert_bez_food_steps": 0}])
def test_invalid_arguments_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SnakeEnv(np.random.default_rng(0), **kwargs)


def test_reset_places_snake_in_center_facing_right() -> None:
    env = SnakeEnv(np.random.default_rng(0))
    assert list(env.snake) == [(7, 7), (7, 6), (7, 5)]
    assert env.direction is Direction.RIGHT
    assert env.score == 0
    assert env.steps_without_food == 0
    assert env.food is not None
    assert not env.is_blocked(env.food)


def test_step_moves_head_forward() -> None:
    env = make_env()
    assert env.step(Action.STRAIGHT) == ALIVE
    assert list(env.snake) == [(7, 8), (7, 7), (7, 6)]
    assert env.steps_without_food == 1


def test_hitting_wall_terminates() -> None:
    env = make_env()
    for _ in range(env.size - 1 - env.head[1]):
        assert env.step(Action.STRAIGHT) == ALIVE
    assert env.step(Action.STRAIGHT) == (REWARD_DEATH, True, False)


def test_hitting_own_body_terminates() -> None:
    env = make_env()
    eat_ahead(env)
    eat_ahead(env)
    assert env.step(Action.LEFT) == ALIVE
    assert env.step(Action.LEFT) == ALIVE
    assert env.step(Action.LEFT) == (REWARD_DEATH, True, False)


def test_head_may_enter_cell_vacated_by_tail() -> None:
    env = make_env()
    eat_ahead(env)
    assert env.step(Action.RIGHT) == ALIVE
    assert env.step(Action.RIGHT) == ALIVE
    tail = env.snake[-1]
    assert env.step(Action.RIGHT) == ALIVE
    assert env.head == tail


def test_eating_grows_snake_and_resets_hunger() -> None:
    env = make_env()
    env.step(Action.STRAIGHT)
    env.food = (7, 9)
    assert env.step(Action.STRAIGHT) == (REWARD_FOOD, False, False)
    assert list(env.snake) == [(7, 9), (7, 8), (7, 7), (7, 6)]
    assert env.score == 1
    assert env.steps_without_food == 0
    assert env.food is not None
    assert not env.is_blocked(env.food)


def test_starving_truncates() -> None:
    env = make_env(chert_bez_food_steps=5)
    for _ in range(4):
        assert env.step(Action.STRAIGHT) == ALIVE
    assert env.step(Action.STRAIGHT) == (REWARD_STEP, False, True)


def test_filling_board_wins() -> None:
    env = SnakeEnv(np.random.default_rng(0), size=4)
    env.snake = deque(
        [
            (3, 1),
            (3, 2),
            (3, 3),
            (2, 3),
            (2, 2),
            (2, 1),
            (2, 0),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (0, 3),
            (0, 2),
            (0, 1),
            (0, 0),
        ]
    )
    env._occupied = set(env.snake)
    env.direction = Direction.LEFT
    env.food = (3, 0)
    assert env.step(Action.STRAIGHT) == (REWARD_FOOD, True, False)
    assert env.food is None
    assert env.score == env.size**2 - INITIAL_LENGTH


def test_random_play_keeps_invariants() -> None:
    rng = np.random.default_rng(1)
    env = SnakeEnv(np.random.default_rng(2))
    for _ in range(200):
        env.reset()
        assert_consistent(env)
        terminated = truncated = False
        while not (terminated or truncated):
            _, terminated, truncated = env.step(Action(rng.integers(len(Action))))
            if not terminated:
                assert_consistent(env)


def test_same_seed_gives_same_game() -> None:
    assert food_sequence(0, 20) == food_sequence(0, 20)


def test_reset_continues_random_sequence() -> None:
    assert len(set(food_sequence(0, 20))) > 1
