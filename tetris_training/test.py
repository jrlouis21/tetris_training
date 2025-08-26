import argparse
import os
import random

import numpy as np
from stable_baselines3 import PPO  # Proximal Policy Optimization Algorithm
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

from .environment import TetrisEnv

GAME_PATH = "./roms/tetris.gb"
os.environ["DISPLAY"] = ":1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Record a video of the best Tetris model playing."
    )
    p.add_argument(
        "--model",
        type=str,
        default="./tetris_training/training_data/best/best_model.zip",
        help="Path to best_model.zip",
    )
    p.add_argument(
        "--record",
        type=bool,
        default=True,
        help="Whether to record video output of the test run.",
    )
    p.add_argument(
        "--video_output_path",
        type=str,
        default="./videos/",
        help="Output video path (mp4)",
    )
    p.add_argument(
        "--video_name", type=str, default=None, help="The name of the saved video"
    )
    p.add_argument("--seed", type=int, default=42, help="Test run seed.")
    return p.parse_args()


def main(
    model_path: str | None = None,
    seed: int | None = None,
    headless: bool = False,
    record_video: bool = True,
    video_output_path: str | None = None,
    video_output_name: str | None = None,
    episodes=1,  # Run the model 5 times
    deterministic=True,  # stochastic if set to false
):
    # Initialize the gym.Env and then check for any errors in the gym Environment before continuing.
    env = TetrisEnv(
        game_path=str(GAME_PATH),
        headless=headless,
        record_video=record_video,
        video_output_path=video_output_path,
        video_output_name=video_output_name,
    )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        env.reset(seed=seed)

    check_env(env, warn=True)

    # Use an existing trained model if it is provided; otherwise will create a new (untrained) model.
    if model_path:
        model = PPO.load(model_path, env=env)
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, seed=seed)

    # Runs the model and provides the mean reward across the number of episodes ran
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=episodes,
        deterministic=deterministic,
    )
    print(
        f"[deterministic={deterministic}] "
        f"mean_reward over {episodes} eps: {mean_reward:.2f} +/- {std_reward:.2f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        model_path=args.model,
        record_video=args.record,
        video_output_path=args.video_output_path,
        video_output_name=args.video_name,
        seed=args.seed,
    )
