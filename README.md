# Baby GPT

Baby GPT is a small, character-level GPT implementation in PyTorch. The
notebook builds the model, trains it on Tiny Shakespeare, generates text, and
then explores how attention width, head count, dropout, depth, residual
connections, and layer normalization affect validation loss.

## Run the notebook

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if it is
not already available, then start JupyterLab from the repository root:

```bash
uv run jupyter lab
```

uv creates the project environment and installs the locked dependencies on the
first run. Open `baby-gpt.ipynb` in JupyterLab and run the cells in order. The
Tiny Shakespeare dataset used by the notebook is included under `data/`.

## Experiment setup

All runs use the same data and training configuration:

- Tiny Shakespeare character-level dataset
- 90/10 train/validation split
- Context length: 256 characters
- Batch size: 32
- Adam optimizer with a learning rate of `1e-3`
- At most 50,000 training steps
- Validation every 1,000 steps
- Early stopping after 10 validation checks without improvement

## Results

The table highlights the milestones from the full experiment sequence in the
notebook.

| Embedding | Attention | Blocks | Dropout | Residual | LayerNorm | Best validation loss |
| --------: | --------: | -----: | ------: | :------: | :-------: | -------------------: |
|       128 |   1 x 512 |      1 |     0.0 |    No    |     No    |               1.7501 |
|       128 |    8 x 32 |      1 |     0.0 |    No    |     No    |               1.6533 |
|       128 |    8 x 32 |      1 |     0.2 |    No    |     No    |               1.6140 |
|       128 |    8 x 32 |      2 |     0.2 |    No    |     No    |               1.5467 |
|       128 |    8 x 32 |      4 |     0.2 |    No    |     No    |               3.3469 |
|       128 |    8 x 32 |      4 |     0.2 |   Yes    |     No    |               1.4930 |
|       128 |    8 x 32 |      4 |     0.2 |   Yes    |    Yes    |               1.4760 |
|       128 |    8 x 32 |      6 |     0.4 |   Yes    |    Yes    |               1.4486 |
|       128 |    8 x 32 |      8 |     0.4 |   Yes    |    Yes    |               1.4463 |

## Conclusions

- **Residual connections were the key architectural change.** A four-block
  model without residuals collapsed at a validation loss of 3.3469. Adding
  residuals reduced it to 1.4930.
- The failure was visible inside the network. Without residuals, 43–54% of MLP
  units never fired at initialization and gradient magnitude varied by 612x
  across blocks. With residuals, the never-fired share fell to 0% and the
  gradient spread fell to 2.8x.
- **Dead and sparse activations are different failure modes.** A model with
  roughly 79% zero activations but no permanently inactive units still trained
  well, while a model with roughly 90% inactive units failed.
- **Depth helped after residual connections were added, but its benefit
  saturated.** The incremental validation-loss improvements were approximately
  0.050, 0.046, 0.014, and 0.001 as blocks were added.
- **LayerNorm was neutral to mildly positive.** Its improvement from 1.4930 to
  1.4760 is within the roughly 0.02 run-to-run noise and is not conclusive.
- Increasing `head_size` beyond the embedding width did not add useful model
  capacity; the apparent benefit in an earlier run was an optimization effect.
- Wider, 256-dimensional embeddings did not improve validation loss in these
  experiments. The available measurements are not enough to conclude that the
  wider models overfit.
- The best run used 128-dimensional embeddings, eight 32-dimensional attention
  heads, eight blocks, 40% dropout, residual connections, and LayerNorm. It
  reached a validation loss of **1.4463**, only slightly better than the
  six-block model's 1.4486.

## Measurement note

Because training uses early stopping, elapsed time measures time until stopping,
not pure architecture cost. Future runs should record milliseconds per step and
the step of the best checkpoint separately.
