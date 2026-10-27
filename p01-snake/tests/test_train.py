import numpy as np
import pytest

from snake.agents import AgentConfig, QLearning
from snake.env import Action, SnakeEnv
from snake.state import N_STATES
from snake.train import TrainConfig, epsilon_at, evaluate, make_rngs, trailing_mean, train


class Recorder:
    def __init__(self) -> None:
        self.steps = 0
        self.episodes: list[tuple[int, int, float]] = []

    def on_step(self, env: SnakeEnv) -> None:
        self.steps += 1

    def on_episode_end(self, episode: int, score: int, epsilon: float) -> None:
        self.episodes.append((episode, score, epsilon))


def make_setup(seed: int) -> tuple[QLearning, SnakeEnv]:
    env_rng, agent_rng = make_rngs(seed)
    return QLearning(N_STATES, len(Action), AgentConfig(), agent_rng), SnakeEnv(env_rng)


def test_epsilon_decays_linearly_then_stays() -> None:
    config = TrainConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_episodes=10)
    assert epsilon_at(config, 0) == pytest.approx(1.0)
    assert epsilon_at(config, 5) == pytest.approx(0.55)
    assert epsilon_at(config, 10) == pytest.approx(0.1)
    assert epsilon_at(config, 1000) == pytest.approx(0.1)


def test_trailing_mean_uses_available_history() -> None:
    np.testing.assert_allclose(trailing_mean(np.array([2, 4, 6, 8]), 2), [2, 3, 5, 7])


def test_train_reports_every_episode_to_callback() -> None:
    agent, env = make_setup(0)
    recorder = Recorder()
    config = TrainConfig(n_episodes=5, epsilon_decay_episodes=4)
    scores = train(agent, env, config, recorder)
    assert [episode for episode, _, _ in recorder.episodes] == list(range(5))
    assert [score for _, score, _ in recorder.episodes] == scores.tolist()
    assert recorder.episodes[-1][2] == pytest.approx(config.epsilon_end)
    assert recorder.steps >= 5


def test_same_seed_gives_same_training() -> None:
    config = TrainConfig(n_episodes=50)
    first, second = make_setup(3), make_setup(3)
    np.testing.assert_array_equal(train(*first, config), train(*second, config))
    np.testing.assert_array_equal(first[0].q, second[0].q)


def test_evaluate_does_not_learn() -> None:
    agent, env = make_setup(0)
    train(agent, env, TrainConfig(n_episodes=20))
    q = agent.q.copy()
    evaluate(agent, env, 5)
    np.testing.assert_array_equal(agent.q, q)


def test_agent_learns_to_eat() -> None:
    agent, env = make_setup(0)
    train(agent, env, TrainConfig(n_episodes=300, epsilon_decay_episodes=100))
    assert evaluate(agent, env, 20).mean() > 5
