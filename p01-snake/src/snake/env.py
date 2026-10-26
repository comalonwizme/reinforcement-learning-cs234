from __future__ import annotations

from collections import deque
from enum import IntEnum

import numpy as np

type Cell = tuple[int, int]

REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
REWARD_STEP = 0.0

INITIAL_LENGTH = 3


class Action(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Cell:
        return _DELTAS[self]

    def turned(self, action: Action) -> Direction:
        return Direction((self + _TURN_OFFSETS[action]) % len(Direction))


_DELTAS: tuple[Cell, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
_TURN_OFFSETS: tuple[int, ...] = (0, -1, 1)


class SnakeEnv:
    def __init__(
        self, rng: np.random.Generator, size: int = 15, chert_bez_food_steps: int = 150
    ) -> None:
        if size < 4:
            raise ValueError(f"size ote az. Must be >= 4, got {size}")
        if chert_bez_food_steps < 1:
            raise ValueError(f"steps without food must be >= 1, got {chert_bez_food_steps}")

        self.size = size
        self.chert_bez_food_steps = chert_bez_food_steps
        self._rng = rng

        self.snake: deque[Cell] = deque()
        self._occupied: set[Cell] = set()
        self.direction = Direction.RIGHT
        self.food: Cell | None = None
        self.steps_without_food = 0

        self.reset()

    @property
    def head(self) -> Cell:
        return self.snake[0]

    @property
    def score(self) -> int:
        return len(self.snake) - INITIAL_LENGTH

    def is_blocked(self, cell: Cell) -> bool:
        r, c = cell
        inside = 0 <= r < self.size and 0 <= c < self.size
        return not inside or cell in self._occupied

    def reset(self) -> None:
        mid = self.size // 2
        self.snake = deque((mid, mid - i) for i in range(INITIAL_LENGTH))
        self._occupied = set(self.snake)
        self.direction = Direction.RIGHT
        self.steps_without_food = 0
        self._spawn_food()

    def step(self, action: Action) -> tuple[float, bool, bool]:
        self.direction = self.direction.turned(action)
        dr, dc = self.direction.delta
        hr, hc = self.head
        new_head = (hr + dr, hc + dc)
        eats = new_head == self.food

        if not eats:
            tail = self.snake.pop()
            self._occupied.remove(tail)
            if self.is_blocked(new_head):
                self.snake.append(tail)
                self._occupied.add(tail)
                return REWARD_DEATH, True, False

        self.snake.appendleft(new_head)
        self._occupied.add(new_head)

        if eats:
            self.steps_without_food = 0
            self._spawn_food()
            return REWARD_FOOD, self.food is None, False

        self.steps_without_food += 1
        truncated = self.steps_without_food >= self.chert_bez_food_steps
        return REWARD_STEP, False, truncated

    def _spawn_food(self) -> None:
        free = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in self._occupied
        ]
        if not free:
            self.food = None
            return
        self.food = free[self._rng.integers(len(free))]
