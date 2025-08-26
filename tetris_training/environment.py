import sys
import traceback
import uuid
from pathlib import Path

import gymnasium as gym
import mediapy as media
import numpy as np
import structlog
from pyboy import PyBoy
from pyboy.utils import WindowEvent

logger = structlog.get_logger(__file__)


class TetrisEnv(gym.Env):
    def __init__(
        self,
        game_path: str,
        action_frames: int = 24,
        max_episode_steps: int = 5_000,
        death_penalty: float = -5.0,
        line_cleared_reward: float = 1.0,  # reward = (Number of lines cleared ** 2) * lines_cleared_reward
        score_increase_reward: float = 0.0,  # reward = amount of score increase * score_increase_reward
        level_increase_reward: float = 10.0,  # reward = levels gained * level_increase_reward
        max_score: float = 1_000_000.0,
        max_lines: float = 500.0,
        max_level: float = 20.0,
        headless: bool = True,
        record_video: bool = False,
        video_output_path: str = "./videos/",
        video_output_name: str = None,
    ):
        """A gymnasium Env for training an agent to play Tetris using PyBoy.

        Args:
            game_path (str): The path to the game '.gb' file.
            action_frames (int, optional): The total number of frames between actions (including the action itself). Defaults to 24.
            max_episode_steps (int, optional): The maximum number of steps for the episode. Defaults to 5_000.
            death_penalty (float, optional): The penalty incurred when reaching the 'Game Over' screen. Defaults to -5.0.
            line_cleared_reward (float, optional): The reward received per line cleared. Defaults to 1.0.
            max_lines (float, optional): Used to normalize the lines value in the training data. Defaults to 500.0.
            max_level (float, optional): Used to normalize the level value in the training data. Defaults to 20.0.
            headless (bool, optional): When running in headless mode, the game display will not be rendered. Defaults to True.
            record_video (bool, optional): Records a video of the episode. Defaults to False.
            video_output_path (str, optional): Set an output path for the video. Defaults to None.
            video_output_name (str, optional): Set a custom name for the saved video. Defaults to None.
        """
        super().__init__()

        self.instance_id = str(uuid.uuid4())[:8]  # A unique instance indicator
        self.total_steps = 0  # Keep track of the total steps taken during each run

        # Viewing / Recording Settings
        self.headless = headless
        self.record_video = record_video
        self.render_screen = self.headless or self.record_video
        self.full_frame_writer = None
        self.video_output_path = video_output_path
        self.video_output_name = video_output_name
        self.matrix_shape = (18, 10)  # Size of the actual game spaces where pieces drop

        # Run settings
        self.action_frames = (
            action_frames  # Number of frames total to wait after each action
        )
        self.max_episode_steps = max_episode_steps
        self.death_penalty = death_penalty
        self.line_cleared_reward = line_cleared_reward
        self.score_increase_reward = score_increase_reward
        self.level_increase_reward = level_increase_reward
        self.max_lines = max_lines
        self.max_score = max_score
        self.max_level = max_level

        # Valid actions the model is able to perform. Includes a "No Operation" None value,
        # so the Model can choose to wait instead of being forced to perform an action each step.
        self.valid_actions = [
            None,  # No Operation
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
        ]
        self.release_actions = [
            None,  # No Operation
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
        ]

        # List of integers the model is able to choose from for valid actions
        self.action_space = gym.spaces.Discrete(len(self.valid_actions))

        self.game_path = game_path
        self._init_pyboy()

        # Start the inital gameplay loop.
        self.tetris.start_game()
        self.pyboy.tick(1, False)  # ensure VRAM is fresh

        # PyBoy includes a Tetris Wrapper (self.tetris), which gives us various stats during each frame
        # such as lines cleared, score, and level. We will also use this to get a 2D Array of
        # The game board.
        self._update_prev_values()
        self.observation_space = gym.spaces.Dict(
            {
                "board": gym.spaces.Box(
                    low=0, high=255, shape=self.matrix_shape, dtype=np.uint8
                ),
                "score": gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
                "lines": gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
                "level": gym.spaces.Box(low=0.0, high=1.0, dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.instance_id = str(uuid.uuid4())[:8]  # A unique instance indicator

        # reset_game accepts an 8-bit integer seed for its internal time divider register
        # seed is manipulated to keep the lowest 8 bits of seed
        if seed:
            seed = seed & 0xFF

        self.tetris.reset_game(timer_div=seed)

        # tick once to settle memory/VRAM before sampling obs and counters
        self.pyboy.tick(1, False)

        # Reset all values from previous run
        self.total_steps = 0
        self._update_prev_values()

        return self._obs_dict(), {}

    def step(self, action: int):
        """A single step performed by the agent. In each step the agent may perform one action."""

        # During the first step, begin recording video
        if self.record_video and self.total_steps == 0:
            self.start_video()

        self._send_action(action)
        self.total_steps += 1
        reward = 0.0

        terminated = self.tetris.game_over()
        if terminated:
            reward += self.death_penalty
            if self.record_video:
                self.stop_video()

        truncated = False
        if (self.max_episode_steps is not None) and (
            self.total_steps >= self.max_episode_steps
        ):
            truncated = True

        reward += self._calculate_reward()

        self._update_prev_values()
        obs = self._obs_dict()

        info = {
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
        }
        if terminated:
            info.update(
                {
                    "lines": self._prev_lines,
                    "score": self._prev_score,
                    "level": self._prev_level,
                    "steps": self.total_steps,
                }
            )
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render a single game frame."""
        frame = self.pyboy.screen.ndarray
        frame = frame[:, :, 0]
        return np.ascontiguousarray(frame.astype(np.uint8))

    def close(self):
        self.pyboy.stop()

    ####################
    # HELPER FUNCTIONS #
    ####################

    def _init_pyboy(self):
        # Initialize the PyBoy Instance
        window = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            self.game_path,
            window=window,
        )
        self.tetris = self.pyboy.game_wrapper

        if self.headless:
            # Run the game at 6x normal speed
            self.pyboy.set_emulation_speed(0)

        # Sets the Game Area Mapping; mapping_minimal will result in a simple Binary mapping (0=empty square, 1=filled square)
        # Can be changed to mapping_compressed which provides greater detail between different tetronimo types.
        self.tetris.game_area_mapping(self.tetris.mapping_minimal, 0)

    def _obs_dict(self) -> dict:
        """Return a normalized dictionary of observation values.

        Score, Lines, and Level are all normalized to a value between 0-1, which is easier for training than arbitrary numbers.
        """
        return {
            "board": np.asarray(self.tetris.game_area(), dtype=np.uint8),
            "score": np.array([self.tetris.score / self.max_score], dtype=np.float32),
            "lines": np.array([self.tetris.lines / self.max_lines], dtype=np.float32),
            "level": np.array([self.tetris.level / self.max_level], dtype=np.float32),
        }

    def _update_prev_values(self) -> None:
        """Update the 'previous' game data values before going to the next step."""
        self._prev_lines = int(self.tetris.lines)
        self._prev_score = int(self.tetris.score)
        self._prev_level = int(self.tetris.level)
        self._prev_board: np.ndarray = self.tetris.game_area()
        self._game_over = bool(self.tetris.game_over())

    def _send_action(self, action: int) -> None:
        """Manage an action being executed."""

        press = self.valid_actions[action]
        release = self.release_actions[action]

        # When no action is performed, progress by the allotted frames then return
        if press is None:
            self.pyboy.tick(self.action_frames, self.render_screen)
            return

        self.pyboy.send_input(press)
        self.pyboy.tick(self.action_frames // 2, self.render_screen)

        self.pyboy.send_input(release)
        self.pyboy.tick(self.action_frames // 2, self.render_screen)

        if self.record_video:
            self.add_video_frame()

    def _calculate_reward(self) -> float:
        """Calculate a reward value based on the performed action."""
        current_lines = self.tetris.lines
        current_score = self.tetris.score
        current_level = self.tetris.level

        lines_reward = (
            max(current_lines - self._prev_lines, 0) ** 2
        ) * self.line_cleared_reward
        score_reward = (
            max(current_score - self._prev_score, 0) * self.score_increase_reward
        )
        level_reward = (
            max(current_level - self._prev_level, 0) * self.level_increase_reward
        )

        return lines_reward + score_reward + level_reward

    ###################
    # VIDEO RECORDING #
    ###################

    def add_video_frame(self):
        self.full_frame_writer.add_image(self.render())

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()

        video_name = (self.video_output_name or "tetris-run-") + str(self.instance_id)

        base_dir = Path(self.video_output_path)
        base_dir.mkdir(exist_ok=True)
        full_name = Path(video_name).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()

    def stop_video(self):
        if self.full_frame_writer is not None:
            try:
                self.full_frame_writer.__exit__(None, None, None)
            except Exception as e:
                print("VideoWriter close failed:", e, file=sys.stderr)
                traceback.print_exc()
            finally:
                self.full_frame_writer = None
