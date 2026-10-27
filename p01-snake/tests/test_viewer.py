import matplotlib.pyplot as plt
import pytest

from snake.agents import AgentConfig, QLearning
from snake.env import Action, SnakeEnv
from snake.state import N_STATES
from snake.train import TrainConfig, make_rngs, train
from snake.viewer import LiveViewer


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_live_viewer_runs_headless() -> None:
    plt.switch_backend("agg")
    env_rng, agent_rng = make_rngs(0)
    env = SnakeEnv(env_rng)
    agent = QLearning(N_STATES, len(Action), AgentConfig(), agent_rng)
    viewer = LiveViewer(env.size, render_every=3, fps=1000, curve_every=1)
    train(agent, env, TrainConfig(n_episodes=3), viewer)
    viewer.hold()
    plt.close("all")
