# GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS

## Unofficial re-implementation of [this paper](https://mathai-iclr.github.io/papers/papers/MATHAI_29_paper.pdf) by Power et al.

Code written by Charlie Snell.

1. Clone the repository, and move into the directory:

```bash
git clone https://github.com/Sea-Snell/grokking.git
cd grokking/
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
Python 3.12.9 (main, Mar 17 2025, 21:36:21) [Clang 20.1.0 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import grokking
```

## Running the code

To roughly re-create Figure 1 in the paper run:

```bash
uv run grokking/scripts/train_grokk.py
```

Running the above command should give curves like this.

![Training and validation accuracy](grokk.png)

Try different operations or learning / architectural hparams by modifying configurations in the `config/` directory. 
I use [Hydra](https://hydra.cc/docs/intro) to handle the configs (see their documentation to learn how to change configs in the commandline etc...).

Training uses [Weights And Biases](https://wandb.ai/home) by default to generate plots in realtime. 
If you would not like to use wandb, just set `wandb.use_wandb=False` in `config/train_grokk.yaml` or as an argument when calling `train_grokk.py`
