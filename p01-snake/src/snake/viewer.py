import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from snake.env import SnakeEnv
from snake.grid import Tile, to_grid
from snake.train import trailing_mean

TILE_COLORS = {
    Tile.EMPTY: "#1d1f27",
    Tile.BODY: "#3fa34d",
    Tile.HEAD: "#b5e655",
    Tile.FOOD: "#e4572e",
}


class LiveViewer:

    def __init__(
        self,
        size: int,
        render_every: int = 50,
        fps: float = 30.0,
        curve_every: int = 10,
        window: int = 100,
    ) -> None:
        self._render_every = render_every
        self._frame_seconds = 1.0 / fps
        self._curve_every = curve_every
        self._window = window
        self._episode = 0
        self._epsilon = 1.0
        self._scores: list[int] = []

        plt.ion()
        self._fig, (board_ax, self._curve_ax) = plt.subplots(
            1, 2, figsize=(12, 5), width_ratios=(1, 1.6), layout="constrained"
        )
        self._board = board_ax.imshow(
            np.zeros((size, size)),
            cmap=ListedColormap([TILE_COLORS[tile] for tile in Tile]),
            vmin=0,
            vmax=len(Tile) - 1,
            interpolation="nearest",
        )
        board_ax.set_xticks([])
        board_ax.set_yticks([])
        self._board_title = board_ax.set_title("")

        (self._points,) = self._curve_ax.plot([], [], ".", ms=3, alpha=0.3, label="score")
        (self._mean,) = self._curve_ax.plot([], [], lw=2, label=f"avg({window})")
        self._curve_ax.set_xlabel("episode")
        self._curve_ax.set_ylabel("score")
        self._curve_ax.grid(alpha=0.3)
        self._curve_ax.legend(loc="upper left")
        self._curve_title = self._curve_ax.set_title("")
        plt.show(block=False)

    def on_step(self, env: SnakeEnv) -> None:
        if self._episode % self._render_every:
            return
        self._board.set_data(to_grid(env))
        self._board_title.set_text(f"episode {self._episode} · score {env.score}")
        self._refresh(self._frame_seconds)

    def on_episode_end(self, episode: int, score: int, epsilon: float) -> None:
        self._scores.append(score)
        self._episode = episode + 1
        self._epsilon = epsilon
        if self._episode % self._curve_every == 0:
            self._draw_curve()
            self._refresh(0.001)

    def hold(self) -> None:
        self._draw_curve()
        plt.ioff()
        plt.show()

    def _draw_curve(self) -> None:
        scores = np.array(self._scores)
        episodes = np.arange(len(scores))
        mean = trailing_mean(scores, self._window)
        self._points.set_data(episodes, scores)
        self._mean.set_data(episodes, mean)
        self._curve_ax.relim()
        self._curve_ax.autoscale_view()
        self._curve_title.set_text(
            f"ε {self._epsilon:.2f} · best {scores.max()} · avg({self._window}) {mean[-1]:.1f}"
        )

    def _refresh(self, seconds: float) -> None:
        self._fig.canvas.draw_idle()
        self._fig.canvas.start_event_loop(seconds)
