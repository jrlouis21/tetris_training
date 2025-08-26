import multiprocessing as mp
import os
import traceback
from pathlib import Path

import structlog
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from .callbacks import TensorboardCallback
from .environment import TetrisEnv

LOGGER = structlog.get_logger(__file__)

SEED = 42
HEADLESS = True

# Set up directory to store training data
RUN_DIRECTORY = Path("./tetris_training/training_data/")
RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RUN_DIRECTORY / "tetris.zip"
VECSCALE_PATH = RUN_DIRECTORY / "vecnormalize.pkl"
GAME_PATH = Path("./roms/tetris.gb")

# Dynamically set the number of CPUs to use for training based on system resources. Up to a maximum of 24 cores.
DEFAULT_ENVS = max(4, min(24, (mp.cpu_count() or 8) // 1))
NUM_CPU = int(os.environ.get("TETRIS_ENVS", DEFAULT_ENVS))


def make_env(rank: int, env_conf: dict, seed: int = 0):
    def _init():
        try:
            # Monitor gives us episode returns/lengths in Infos and for SB3 logs
            env = Monitor(TetrisEnv(**env_conf))
            env.reset(seed=seed + rank)
            return env
        except Exception as e:
            LOGGER.error("Worker failed to create env", rank=rank, exception=e)
            traceback.print_exc()
            raise

    return _init


def build_vecenv(n_envs: int, env_conf: dict, seed: int) -> SubprocVecEnv:
    return SubprocVecEnv(
        [make_env(i, env_conf, seed) for i in range(n_envs)], start_method="spawn"
    )


def main():
    if not GAME_PATH.exists():
        raise FileNotFoundError(f"ROM not found at {GAME_PATH.resolve()}")

    set_random_seed(SEED)

    env_config = dict(
        game_path=str(GAME_PATH),
        headless=HEADLESS,
        max_episode_steps=2_000,
    )

    train_env = build_vecenv(NUM_CPU, env_config, SEED)

    # Reward/obs normalization greatly helps PPO on sparse/peaky rewards.
    # Important: we save & reload stats across runs.
    if VECSCALE_PATH.exists():
        train_env = VecNormalize.load(str(VECSCALE_PATH), train_env)
        train_env.training = True
        # keep gamma consistent with PPO gamma
        train_env.gamma = 0.997
    else:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,  # safe with "binary"/"float32" obs
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.997,
        )

    # Use DummyVecEnv with 1–4 envs; share VecNormalize stats.
    eval_env_raw = DummyVecEnv(
        [make_env(10_000 + i, env_config, SEED) for i in range(2)]
    )
    if VECSCALE_PATH.exists():
        eval_env = VecNormalize.load(str(VECSCALE_PATH), eval_env_raw)
    else:
        # create a fresh wrapper that will receive stats from train wrapper
        eval_env = VecNormalize(
            eval_env_raw,
            training=False,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=0.997,
        )
    eval_env.training = False  # VERY important: do not update stats during eval

    # ---- rollout / batch sizing ----
    # Target ~128k transitions/update. Adjust n_steps from n_envs.
    TARGET_ROLLOUT = 128_000
    n_steps = max(128, TARGET_ROLLOUT // NUM_CPU)  # per-env
    # Bigger batch & a couple epochs for PPO stability on larger buffers.
    batch_size = 4096
    n_epochs = 2

    LOGGER.info(
        "trainer-shape",
        n_envs=NUM_CPU,
        n_steps=n_steps,
        rollout_size=n_steps * NUM_CPU,
        batch_size=batch_size,
        n_epochs=n_epochs,
    )

    # ---- callbacks ----
    checkpoint_callback = CheckpointCallback(
        save_freq=(n_steps * NUM_CPU) // 2,  # every ~half update
        save_path=str(RUN_DIRECTORY),
        name_prefix="tetris",
        save_replay_buffer=False,
        save_vecnormalize=True,  # SB3 will save a copy alongside
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(RUN_DIRECTORY / "best"),
        log_path=str(RUN_DIRECTORY / "eval"),
        eval_freq=n_steps * NUM_CPU,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    tensorboard_callback = TensorboardCallback(train_env)

    # ---- model (create or load) ----
    policy_kwargs = dict(net_arch=[256, 256])
    if MODEL_PATH.exists():
        LOGGER.info("Loading existing PPO", file=str(MODEL_PATH))
        model = PPO.load(
            str(MODEL_PATH),
            env=train_env,
            device="auto",
            custom_objects={
                # let us tweak a few hparams on resume
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": 0.997,
                "ent_coef": 0.01,
            },
        )
        # If the saved model had different policy arch, we keep it; change later if needed.
    else:
        model = PPO(
            "MultiInputPolicy",
            train_env,
            verbose=1,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.997,
            ent_coef=0.01,
            learning_rate=3e-4,
            seed=SEED,
            tensorboard_log=str(RUN_DIRECTORY / "tboard"),
            policy_kwargs=policy_kwargs,
            device="auto",
        )

    # ---- train ----
    # SB3 counts timesteps across all envs (i.e., 1 call to .step() adds n_envs to counter)
    total_timesteps = 500_000_000  # start sane; scale once you see learning
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(
            [checkpoint_callback, eval_callback, tensorboard_callback]
        ),
        progress_bar=True,
    )

    # Save model and VecNormalize stats
    model.save(str(MODEL_PATH))
    train_env.save(str(VECSCALE_PATH))  # persist running mean/var

    # Close up
    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    LOGGER.info("Beginning new session", n_envs=NUM_CPU)
    main()
