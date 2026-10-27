from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgentConfig:
    alpha: float = 0.1
    gamma: float = 0.9


def _greedy(values: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(values == values.max())))


def _epsilon_greedy(values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(values)))
    return _greedy(values, rng)


class QLearning:
    def __init__(
        self, n_states: int, n_actions: int, config: AgentConfig, rng: np.random.Generator
    ) -> None:
        self.q = np.zeros((n_states, n_actions))
        self._config = config
        self._rng = rng

    def values(self, state: int) -> np.ndarray:
        return self.q[state]

    def act(self, state: int, epsilon: float) -> int:
        return _epsilon_greedy(self.values(state), epsilon, self._rng)

    def update(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        target = reward
        if not terminated:
            target += self._config.gamma * self.q[next_state].max()
        self.q[state, action] += self._config.alpha * (target - self.q[state, action])


class DoubleQLearning:
    def __init__(
        self, n_states: int, n_actions: int, config: AgentConfig, rng: np.random.Generator
    ) -> None:
        self.q1 = np.zeros((n_states, n_actions))
        self.q2 = np.zeros((n_states, n_actions))
        self._config = config
        self._rng = rng

    def values(self, state: int) -> np.ndarray:
        return self.q1[state] + self.q2[state]

    def act(self, state: int, epsilon: float) -> int:
        return _epsilon_greedy(self.values(state), epsilon, self._rng)

    def update(
        self, state: int, action: int, reward: float, next_state: int, terminated: bool
    ) -> None:
        learner, evaluator = (self.q1, self.q2) if self._rng.random() < 0.5 else (self.q2, self.q1)
        target = reward
        if not terminated:
            best = int(np.argmax(learner[next_state]))
            target += self._config.gamma * evaluator[next_state, best]
        learner[state, action] += self._config.alpha * (target - learner[state, action])

AGENTS: dict[str, type[QLearning] | type[DoubleQLearning]] = {
    "q": QLearning,
    "double-q": DoubleQLearning,
}