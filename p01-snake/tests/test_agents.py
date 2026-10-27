import numpy as np
import pytest

from snake.agents import AgentConfig, DoubleQLearning, QLearning

CONFIG = AgentConfig(alpha=0.5, gamma=0.9)


@pytest.fixture(params=[QLearning, DoubleQLearning])
def agent(request: pytest.FixtureRequest) -> QLearning | DoubleQLearning:
    return request.param(4, 3, CONFIG, np.random.default_rng(0))


def test_q_update_bootstraps_from_best_next_action() -> None:
    agent = QLearning(4, 3, CONFIG, np.random.default_rng(0))
    agent.q[2] = [0.0, 5.0, 1.0]
    agent.update(0, 1, 1.0, 2, terminated=False)
    assert agent.q[0, 1] == pytest.approx(0.5 * (1.0 + 0.9 * 5.0))


def test_q_update_does_not_bootstrap_on_termination() -> None:
    agent = QLearning(4, 3, CONFIG, np.random.default_rng(0))
    agent.q[2] = [0.0, 5.0, 1.0]
    agent.update(0, 1, -10.0, 2, terminated=True)
    assert agent.q[0, 1] == pytest.approx(-5.0)


def test_double_q_updates_one_table_evaluated_by_the_other() -> None:
    seen = set()
    for seed in range(20):
        agent = DoubleQLearning(4, 3, CONFIG, np.random.default_rng(seed))
        agent.q1[2] = [0.0, 10.0, 0.0]
        agent.q2[2] = [3.0, 1.0, 7.0]
        agent.update(0, 1, 1.0, 2, terminated=False)
        if agent.q1[0, 1]:
            assert agent.q1[0, 1] == pytest.approx(0.5 * (1.0 + 0.9 * 1.0))
            assert agent.q2[0, 1] == 0.0
            seen.add("q1")
        else:
            assert agent.q2[0, 1] == pytest.approx(0.5 * (1.0 + 0.9 * 0.0))
            seen.add("q2")
    assert seen == {"q1", "q2"}


def test_double_q_acts_on_sum_of_tables() -> None:
    agent = DoubleQLearning(4, 3, CONFIG, np.random.default_rng(0))
    agent.q1[0] = [1.0, 0.0, 0.0]
    agent.q2[0] = [0.0, 0.0, 2.0]
    np.testing.assert_allclose(agent.values(0), [1.0, 0.0, 2.0])
    assert agent.act(0, epsilon=0.0) == 2


def test_greedy_breaks_ties_randomly(agent: QLearning | DoubleQLearning) -> None:
    assert {agent.act(0, epsilon=0.0) for _ in range(100)} == {0, 1, 2}


def test_greedy_picks_best_action(agent: QLearning | DoubleQLearning) -> None:
    agent.update(0, 2, 1.0, 1, terminated=True)
    agent.update(0, 2, 1.0, 1, terminated=True)
    assert {agent.act(0, epsilon=0.0) for _ in range(100)} == {2}


def test_full_exploration_tries_every_action(agent: QLearning | DoubleQLearning) -> None:
    agent.update(0, 2, 1.0, 1, terminated=True)
    agent.update(0, 2, 1.0, 1, terminated=True)
    assert {agent.act(0, epsilon=1.0) for _ in range(100)} == {0, 1, 2}
