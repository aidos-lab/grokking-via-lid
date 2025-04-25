# Private fork of the repository [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking)

The original code, written by Charlie Snell, is an unofficial re-implementation of the paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) by Power et al.
This is a private fork of the repository [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking).

This has been extended to allow the computation of topological local estimates of the language model's hidden states during the training process.

## General setup

1. Clone the repository, and move into the directory:

```bash
git clone https://github.com/ben300694/grokking-private
cd grokking-private/
```

1. This package works with `uv`. If you already have `uv` installed, you can create a new environment as follows:

```bash
uv lock
uv sync
```

1. Now, you can directly run the training script as indicated below via `uv run`.
If you would like to use the package in another way, you can start a python interpreter in the environment:

```bash
uv run python3
```

In the interpreter, you can import the package as follows:

```bash
Python 3.12.9 (main, Mar 17 2025, 21:36:21) [Clang 20.1.0] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import grokking
```

## Project-specific setup

1. Set the correct environment variables used in the project config.
Edit the script `grokking/scripts/setup_environment.sh` with the correct paths and run it once.

```bash
./grokking/scripts/setup_environment.sh
```

1. If required, e.g. when running jobs on the HHU Hilbert HPC cluster, set the correct environment variables in the `.env` file in the project root directory.

1. For setting up the repository to support job submissions to the HHU Hilbert HPC, follow the instructions here: [https://github.com/carelvniekerk/Hydra-HPC-Launcher].

## Running the code

To roughly re-create Figure 1 in the paper run:

```bash
uv run grokking/scripts/train_grokk.py
```

Running the above command should give curves like this.

![Training and validation accuracy](grokk.png)

Some `uv run` commands are defined in the `pyproject.toml` file, which can be used as entry points to run the code.
These also accept command line arguments, so for example, for running the training with a larger training fraction and without wandb, you can run:

```bash
uv run train_grokk dataset.frac_train=0.5 wandb.use_wandb=false
```

Try different operations or learning / architectural hparams by modifying configurations in the `config/` directory.
This package uses [Hydra](https://hydra.cc/docs/intro) to handle the configs (see their documentation to learn how to change configs in the commandline etc ...).

Training uses [Weights And Biases](https://wandb.ai/home) by default to generate plots in realtime.
If you would not like to use wandb, just set `wandb.use_wandb=False` in `config/train_grokk.yaml` or as an argument when calling `train_grokk.py`
