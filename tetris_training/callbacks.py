# episode_tb_callback.py
from __future__ import annotations

from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat


class TensorboardCallback(BaseCallback):
    """
    Log one scalar per finished episode (per env) directly to TensorBoard,
    using an internal episode counter as global_step so you get ALL points.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.tb_writer = None
        self._total_episode_count = 0
        self._per_env_ep_len: Optional[np.ndarray] = None  # fallback if no Monitor

    def _on_training_start(self) -> None:
        # find the TensorBoard writer SB3 created (requires tensorboard_log=... in model)
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self.tb_writer = fmt.writer
                break
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._per_env_ep_len = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        dones: Optional[np.ndarray] = self.locals.get("dones")
        infos: Optional[List[dict]] = self.locals.get("infos")
        if dones is None or infos is None:
            return True

        self._per_env_ep_len += 1  # fallback step counts

        for i, done in enumerate(dones):
            if not bool(done):
                continue

            info = infos[i] or {}
            ep = info.get("episode")  # present if env is wrapped with Monitor

            # Gather metrics (prefer your info, then Monitor, then fallback)
            lines = info.get("lines", np.nan)
            score = info.get("score", np.nan)
            level = info.get("level", np.nan)
            steps = info.get(
                "steps",
                (
                    ep.get("l")
                    if isinstance(ep, dict) and "l" in ep
                    else self._per_env_ep_len[i]
                ),
            )

            # Write a point per metric with episode index as global_step
            if self.tb_writer is not None:
                step = int(self._total_episode_count)
                if not np.isnan(lines):
                    self.tb_writer.add_scalar("env_runs/lines", float(lines), step)
                if not np.isnan(score):
                    self.tb_writer.add_scalar("env_runs/score", float(score), step)
                if not np.isnan(level):
                    self.tb_writer.add_scalar("env_runs/level", float(level), step)
                self.tb_writer.add_scalar("env_runs/steps", float(steps), step)

                # Optional: also log Monitor summaries if you want
                if isinstance(ep, dict):
                    if "r" in ep:
                        self.tb_writer.add_scalar(
                            "env_runs/ep_return", float(ep["r"]), step
                        )
                    if "l" in ep:
                        self.tb_writer.add_scalar(
                            "env_runs/ep_length", float(ep["l"]), step
                        )

                self.tb_writer.flush()

            # reset counter for that env and advance global episode counter
            self._per_env_ep_len[i] = 0
            self._total_episode_count += 1

        return True
