from collections import deque
from enum import IntEnum

import numpy as np

type Cell = tuple[int, int]

DIRS: tuple[Cell, ...] = ((1, 0), (0, 1), (-1, 0), (0, -1))


class Action(IntEnum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


TURNS = (0, 1, -1)


class SnakeEnv:
    REWARD_FOOD = 10.0
    REWARD_DEATH = -10.0
    REWARD_STEP = 0.0

    def __init__(self, width: int = 32, height: int = 24, starve_factor: int = 100) -> None:
        self.width = width
        self.height = height
        self.starve_factor = starve_factor
        self._rng = np.random.default_rng()
        self.reset()

    @property
    def head(self) -> Cell:
        return self._body[0]

    @property
    def body(self) -> tuple[Cell, ...]:
        return tuple(self._body)

    @property
    def direction(self) -> int:
        return self._direction

    @property
    def food(self) -> Cell | None:
        return self._food

    @property
    def score(self) -> int:
        return self._score

    def reset(self, seed: int | np.random.SeedSequence | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        x, y = self.width // 2, self.height // 2
        self._set_state([(x, y), (x - 1, y), (x - 2, y)], 0, None)
        self._food = self._place_food()

    def next_cell(self, action: int) -> Cell:
        dx, dy = DIRS[self._turn(action)]
        x, y = self.head
        return x + dx, y + dy

    def would_collide(self, cell: Cell) -> bool:
        x, y = cell
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return cell in self._occupied and cell != self._body[-1]

    def step(self, action: int) -> tuple[float, bool, bool]:
        if self._over:
            raise RuntimeError("episode is over, call reset()")

        new_head = self.next_cell(action)
        self._direction = self._turn(action)

        if self.would_collide(new_head):
            self._over = True
            return self.REWARD_DEATH, True, False

        if new_head == self._food:
            self._body.appendleft(new_head)
            self._occupied.add(new_head)
            self._score += 1
            self._steps_since_food = 0
            self._food = self._place_food()
            if self._food is None:
                self._over = True
                return self.REWARD_FOOD, True, False
            return self.REWARD_FOOD, False, False

        tail = self._body.pop()
        self._occupied.remove(tail)
        self._body.appendleft(new_head)
        self._occupied.add(new_head)
        self._steps_since_food += 1

        if self._steps_since_food > self.starve_factor * len(self._body):
            self._over = True
            return self.REWARD_STEP, False, True
        return self.REWARD_STEP, False, False

    def _turn(self, action: int) -> int:
        return (self._direction + TURNS[action]) % 4

    def _place_food(self) -> Cell | None:
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in self._occupied
        ]
        if not free:
            return None
        return free[self._rng.integers(len(free))]

    def _set_state(self, body: list[Cell], direction: int, food: Cell | None) -> None:
        self._body = deque(body)
        self._occupied = set(body)
        self._direction = direction
        self._food = food
        self._score = 0
        self._steps_since_food = 0
        self._over = False
