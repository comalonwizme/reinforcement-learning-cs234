from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from snake.agents import AGENTS, AgentConfig
from snake.env import Action, SnakeEnv
from snake.state import N_STATES
from snake.train import TrainConfig, evaluate, make_rngs, trailing_mean, train

N_SEEDS = 5
EVAL_EPISODES = 100
WINDOW = 100
RESULTS = Path(__file__).resolve().parents[1] / "results"


def run(agent_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    env_rng, agent_rng = make_rngs(seed)
    env = SnakeEnv(env_rng)
    agent = AGENTS[agent_name](N_STATES, len(Action), AgentConfig(), agent_rng)
    return train(agent, env, TrainConfig()), evaluate(agent, env, EVAL_EPISODES)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.subplots()
    rows = []
    for name in AGENTS:
        runs = [run(name, seed) for seed in range(N_SEEDS)]
        curves = np.array([trailing_mean(scores, WINDOW) for scores, _ in runs])
        greedy = np.array([eval_scores.mean() for _, eval_scores in runs])
        best = max(int(eval_scores.max()) for _, eval_scores in runs)

        episodes = np.arange(curves.shape[1])
        mean, std = curves.mean(axis=0), curves.std(axis=0)
        ax.plot(episodes, mean, label=name)
        ax.fill_between(episodes, mean - std, mean + std, alpha=0.2)
        rows.append(f"| {name} | {greedy.mean():.1f} ± {greedy.std():.1f} | {best} |")

    ax.set_xlabel("episode")
    ax.set_ylabel(f"score, avg({WINDOW})")
    ax.set_title(f"Training score, mean ± std over {N_SEEDS} seeds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(RESULTS / "learning_curves.png", dpi=150)

    table = "\n".join(
        [
            f"| agent | greedy score over {EVAL_EPISODES} games | best game |",
            "|---|---|---|",
            *rows,
        ]
    )
    (RESULTS / "summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
