import argparse

from snake.agents import AGENTS, AgentConfig
from snake.env import Action, SnakeEnv
from snake.state import N_STATES
from snake.train import TrainConfig, evaluate, make_rngs, train
from snake.viewer import LiveViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an agent offline, then watch it play.")
    parser.add_argument("--agent", choices=AGENTS, default="q")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    env_rng, agent_rng = make_rngs(args.seed)
    env = SnakeEnv(env_rng)
    agent = AGENTS[args.agent](N_STATES, len(Action), AgentConfig(), agent_rng)
    train(agent, env, TrainConfig())
    viewer = LiveViewer(env.size, render_every=1, fps=args.fps, curve_every=1)
    evaluate(agent, env, args.games, viewer)
    viewer.hold()


if __name__ == "__main__":
    main()
