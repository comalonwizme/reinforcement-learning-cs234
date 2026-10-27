import argparse

from snake.agents import AGENTS, AgentConfig
from snake.env import Action, SnakeEnv
from snake.state import N_STATES
from snake.train import TrainConfig, make_rngs, train
from snake.viewer import LiveViewer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an agent and watch it learn live.")
    parser.add_argument("--agent", choices=AGENTS, default="q")
    parser.add_argument("--episodes", type=int, default=TrainConfig.n_episodes)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-every", type=int, default=50)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    env_rng, agent_rng = make_rngs(args.seed)
    env = SnakeEnv(env_rng)
    agent = AGENTS[args.agent](N_STATES, len(Action), AgentConfig(), agent_rng)
    viewer = LiveViewer(env.size, render_every=args.render_every, fps=args.fps)
    train(agent, env, TrainConfig(n_episodes=args.episodes), viewer)
    viewer.hold()


if __name__ == "__main__":
    main()
