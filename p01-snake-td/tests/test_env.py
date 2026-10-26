import copy

import numpy as np
import pytest

from snake_td.env import Action, SnakeEnv

CENTER_BODY = [(16, 12), (15, 12), (14, 12)]
FAR_FOOD = (0, 0)


def make_env(body, direction, food, **kwargs) -> SnakeEnv:
    env = SnakeEnv(**kwargs)
    env._set_state(body, direction, food)
    return env


def test_reset():
    env = SnakeEnv()
    env.reset(seed=0)
    assert env.body == ((16, 12), (15, 12), (14, 12))
    assert env.head == (16, 12)
    assert env.direction == 0
    assert env.score == 0
    assert env.food is not None
    assert env.food not in env.body


def test_right_from_r_turns_down():
    env = make_env(CENTER_BODY, 0, FAR_FOOD)
    env.step(Action.RIGHT)
    assert env.direction == 1
    assert env.head == (16, 13)


def test_left_from_r_turns_up():
    env = make_env(CENTER_BODY, 0, FAR_FOOD)
    env.step(Action.LEFT)
    assert env.direction == 3
    assert env.head == (16, 11)


def test_four_rights_restore_direction():
    env = make_env(CENTER_BODY, 0, FAR_FOOD)
    for _ in range(4):
        _, terminated, truncated = env.step(Action.RIGHT)
        assert not terminated
        assert not truncated
    assert env.direction == 0


def test_straight_keeps_direction():
    env = make_env(CENTER_BODY, 0, FAR_FOOD)
    env.step(Action.STRAIGHT)
    assert env.direction == 0
    assert env.head == (17, 12)


def test_wall():
    env = make_env([(31, 12), (30, 12), (29, 12)], 0, FAR_FOOD)
    assert env.step(Action.STRAIGHT) == (SnakeEnv.REWARD_DEATH, True, False)


def test_food():
    env = make_env(CENTER_BODY, 0, (17, 12))
    reward, terminated, truncated = env.step(Action.STRAIGHT)
    assert reward == SnakeEnv.REWARD_FOOD
    assert not terminated
    assert not truncated
    assert len(env.body) == 4
    assert env.score == 1
    assert env.head == (17, 12)
    assert env.food is not None
    assert env.food not in env.body


def test_filling_board_is_victory():
    env = SnakeEnv(width=4, height=1)
    env.reset(seed=0)
    assert env.food == (3, 0)
    assert env.step(Action.STRAIGHT) == (SnakeEnv.REWARD_FOOD, True, False)
    assert env.food is None
    assert env.score == 1


def test_snake_chasing_own_tail_survives():
    env = make_env([(1, 0), (0, 0), (0, 1), (1, 1)], 0, (10, 10), starve_factor=1000)
    for _ in range(1000):
        assert env.next_cell(Action.RIGHT) == env.body[-1]
        assert env.step(Action.RIGHT) == (SnakeEnv.REWARD_STEP, False, False)
        assert len(env.body) == 4
        assert env.would_collide(env.body[1])


def test_moving_into_body_not_tail_is_death():
    body = [(1, 0), (0, 0), (0, 1), (1, 1), (2, 1)]
    env = make_env(body, 0, (10, 10))
    assert env.step(Action.RIGHT) == (SnakeEnv.REWARD_DEATH, True, False)
    assert env.body == tuple(body)


def test_starvation_truncates_on_exact_step():
    env = make_env(CENTER_BODY, 0, FAR_FOOD, starve_factor=1)
    for _ in range(3):
        assert env.step(Action.STRAIGHT) == (SnakeEnv.REWARD_STEP, False, False)
    assert env.step(Action.STRAIGHT) == (SnakeEnv.REWARD_STEP, False, True)


def food_sequence(seed: int) -> list:
    env = SnakeEnv()
    env.reset(seed=seed)
    actions = np.random.default_rng(0).integers(3, size=500)
    foods = [env.food]
    for action in actions:
        _, terminated, truncated = env.step(int(action))
        if terminated or truncated:
            env.reset()
        foods.append(env.food)
    return foods


def test_same_seed_gives_same_food():
    assert food_sequence(7) == food_sequence(7)
    assert food_sequence(7) != food_sequence(8)


def test_would_collide_matches_step():
    env = SnakeEnv(width=8, height=8)
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(2000):
        for action in Action:
            probe = copy.deepcopy(env)
            _, terminated, _ = probe.step(action)
            assert env.would_collide(env.next_cell(action)) == terminated
        _, terminated, truncated = env.step(int(rng.integers(3)))
        if terminated or truncated:
            env.reset()


def test_step_after_terminated_raises():
    env = make_env([(31, 12), (30, 12), (29, 12)], 0, FAR_FOOD)
    env.step(Action.STRAIGHT)
    with pytest.raises(RuntimeError):
        env.step(Action.STRAIGHT)


def test_step_after_truncated_raises():
    env = make_env(CENTER_BODY, 0, FAR_FOOD, starve_factor=1)
    for _ in range(4):
        env.step(Action.STRAIGHT)
    with pytest.raises(RuntimeError):
        env.step(Action.STRAIGHT)
