# DCGAN

This folder contains the implementation and training scripts for Deep Convolutional GAN (DCGAN) models used to generate 64×64 anime character faces.

## Notebooks

- `dcgan_40k.ipynb`: Trains DCGAN using the Kaggle dataset `soumikrakshit/anime-faces` (~40k images).
- `dcgan_135k.ipynb`: Trains DCGAN using the Kaggle dataset `ziggykor/anime-face-dataset-expanded-2024` (~135k images).

## Outputs

- `models/`: Contains saved model checkpoints for each dataset.
- `generated_images/`: Stores sample images generated after each epoch for visual monitoring of training progress.
