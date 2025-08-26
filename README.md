# Tetris Training (Game Boy) — Gymnasium + Stable-Baselines3 + PyBoy

Train reinforcement learning agents to play **Tetris for Game Boy** using
[Gymnasium](https://gymnasium.farama.org/), [Stable-Baselines3](https://stable-baselines3.readthedocs.io/),
and the [PyBoy](https://docs.pyboy.dk/) emulator.

> This repo is a learning/experiment sandbox focused on reproducible SB3 training,
> parallel environments, and normalized rewards/observations.

---

## Getting Started
### Obtaining a ROM
Before training the model, you will need to have legally obtained a ROM of the Tetris game. The file should have a suffix of `.gb`. Once you have acquired this file you will need to move the file and name it to match the path `./roms/tetris.gb`.

### Running the Model (WIP)
```bash
# 1) install project dependencies
uv sync

# 2) train
uv run -m tetris_training.train

# 3) watch logs
tensorboard --logdir ./tetris_v2/training_data/tboard
```

## References
- [Gymnasium](https://gymnasium.farama.org) - An API standard for reinforcement learning with a diverse collection of reference environments.
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - a set of reliable implementations of reinforcement learning algorithms in PyTorch.
- [PyBoy](https://docs.pyboy.dk) - A Game Boy emulator written in Python.
- [uv](https://github.com/astral-sh/uv) - An extremely fast Python package and project manager, written in Rust.

## License
MIT. See [LICENSE](LICENSE)