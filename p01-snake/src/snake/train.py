from dataclasses import dataclass
from typing import Protocol

import numpy as np

from snake.env import Action, SnakeEnv
from snake.state import encode


class Agent(Protocol):
    def act(self, state: int, epsilon: float) -> int: ...

    def update(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None: ...


class Callback(Protocol):
    def on_step(self, env: SnakeEnv) -> None: ...

    def on_episode_end(self, episode: int, score: int, epsilon: float) -> None: ...


@dataclass(frozen=True)
class TrainConfig:
    n_episodes: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 300


def epsilon_at(config: TrainConfig, episode: int) -> float:
    fraction = min(episode / config.epsilon_decay_episodes, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def make_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    env_seed, agent_seed = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(env_seed), np.random.default_rng(agent_seed)


def run_episode(
    agent: Agent, env: SnakeEnv, epsilon: float, learn: bool, callback: Callback | None
) -> int:
    """Truncation still bootstraps: the snake was cut off, not killed."""
    env.reset()
    state = encode(env)
    terminated = truncated = False
    while not (terminated or truncated):
        action = agent.act(state, epsilon)
        reward, terminated, truncated = env.step(Action(action))
        next_state = encode(env)
        if learn:
            agent.update(state, action, reward, next_state, terminated)
        state = next_state
        if callback is not None:
            callback.on_step(env)
    return env.score


def train(
    agent: Agent, env: SnakeEnv, config: TrainConfig, callback: Callback | None = None
) -> np.ndarray:
    scores = np.zeros(config.n_episodes, dtype=np.int64)
    for episode in range(config.n_episodes):
        epsilon = epsilon_at(config, episode)
        scores[episode] = run_episode(agent, env, epsilon, learn=True, callback=callback)
        if callback is not None:
            callback.on_episode_end(episode, int(scores[episode]), epsilon)
    return scores


def evaluate(
    agent: Agent, env: SnakeEnv, n_episodes: int, callback: Callback | None = None
) -> np.ndarray:
    scores = np.zeros(n_episodes, dtype=np.int64)
    for episode in range(n_episodes):
        scores[episode] = run_episode(agent, env, epsilon=0.0, learn=False, callback=callback)
        if callback is not None:
            callback.on_episode_end(episode, int(scores[episode]), 0.0)
    return scores


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    sums = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return (sums[ends] - sums[starts]) / (ends - starts)
