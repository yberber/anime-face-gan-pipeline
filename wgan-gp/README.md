# WGAN-GP

This folder contains code and training notebooks for Wasserstein GAN with Gradient Penalty (WGAN-GP), used to generate 64×64 anime faces.

## Notebooks

- `wgan-gp_40k.ipynb`: Trains WGAN-GP on `soumikrakshit/anime-faces` (~40k images).
- `wgan-gp_135k.ipynb`: Trains WGAN-GP on `ziggykor/anime-face-dataset-expanded-2024` (~135k images).

## Outputs

- `models/`: Checkpoints for each training run.
- `generated_images/`: Per-epoch sample outputs for visual evaluation.
