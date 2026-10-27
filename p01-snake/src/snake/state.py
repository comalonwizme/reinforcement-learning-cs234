from snake.env import Action, SnakeEnv

N_DANGER = 2 ** len(Action)
N_FOOD = 3 * 3
N_STATES = N_DANGER * N_FOOD

FOOD_NONE = N_FOOD // 2


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def danger(env: SnakeEnv) -> int:
    hr, hc = env.head
    mask = 0
    for action in Action:
        dr, dc = env.direction.turned(action).delta
        if env.is_blocked((hr + dr, hc + dc)):
            mask |= 1 << action
    return mask


def food_position(env: SnakeEnv) -> int:
    if env.food is None:
        return FOOD_NONE

    hr, hc = env.head
    fr, fc = env.food
    dr, dc = fr - hr, fc - hc

    ar, ac = env.direction.delta
    rr, rc = env.direction.turned(Action.RIGHT).delta

    ahead = dr * ar + dc * ac
    side = dr * rr + dc * rc
    return (_sign(ahead) + 1) * 3 + (_sign(side) + 1)


def encode(env: SnakeEnv) -> int:
    return danger(env) * N_FOOD + food_position(env)
