# 🎨 Anime Face GAN Pipeline

This project explores a **two-stage GAN pipeline** to generate high-quality synthetic anime character faces in a resource-efficient way — suitable for training and inference on personal computers (e.g. MacBooks with Apple M1/M2 chips).

Our approach combines:
1. **Image generation** using DCGAN and WGAN-GP architectures trained on anime face datasets.
2. **Image upscaling** using ESRGAN-style super-resolution models (custom-trained variants of Real-ESRGAN Anime).

---

## 🧠 Overview of Trained Models

| Model Name              | Description                          | Weights Location |
|-------------------------|--------------------------------------|------------------|
| **DCGAN-135k**          | DCGAN trained on 135k anime faces    | `dcgan/models/ziggykor_anime-face-dataset-expanded-2024/` |
| **DCGAN-40k**           | DCGAN trained on 40k anime faces     | `dcgan/models/soumikrakshit_anime-faces/` |
| **WGAN-GP-135k**        | WGAN-GP trained on 135k anime faces  | `wgan-gp/models/ziggykor_anime-face-dataset-expanded-2024/` |
| **WGAN-GP-40k**         | WGAN-GP trained on 40k anime faces   | `wgan-gp/models/soumikrakshit_anime-faces/` |
| **ESRGAN V1 (2×)**      | Super-resolution (2×)                | `esrgan_imitation_v1/models/final_generator_2x.pth` |
| **ESRGAN V1 (4×)**      | Super-resolution (4×)                | `esrgan_imitation_v1/models/final_generator_4x.pth` |
| **ESRGAN V2 (4×)**      | Improved super-resolution (4×)       | `esrgan_imitation_v2/models/final_generator_4x.pth` |


---

## 📁 Project Structure

```
anime-face-gan-pipeline/
├── dcgan/                  # DCGAN training notebooks + model checkpoints
├── wgan-gp/                # WGAN-GP training notebooks + model checkpoints
├── esrgan_imitation_v1/    # ESRGAN V1 (2x & 4x) training notebooks + weights
├── esrgan_imitation_v2/    # ESRGAN V2 (4x improved) training + weights
├── real-esrgan/            # Setup for Real-ESRGAN pretrained model
├── sample_images/          # Example low-res and enhanced-downscaled images
├── experiment_images/      # Visual results from our experiments
├── demo.ipynb              # 💡 End-to-end notebook: generate + upscale faces
└── README.md               # You're here!
```


---

## 🚀 Quick Start

### 🛠️ Environment Setup

Install Python dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

> 💡 For MacBooks with M1/M2 chips: Ensure you're using a PyTorch version with MPS support (`torch.device("mps")`).

---

### 🔍 Run the Demo

You can try the whole pipeline (GAN + ESRGAN) using:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yberber/anime-face-gan-pipeline/blob/main/demo-colab.ipynb)

Or run locally:

```bash
jupyter notebook demo.ipynb
```

---

## 📓 Notebook Guide

Each training notebook is located in its corresponding folder:

| Notebook Path                                      | Purpose |
|----------------------------------------------------|---------|
| `dcgan/dcgan_40k.ipynb`                            | Train DCGAN on 40k images |
| `dcgan/dcgan_135k.ipynb`                           | Train DCGAN on 135k images |
| `wgan-gp/wgan-gp_40k.ipynb`                        | Train WGAN-GP on 40k images |
| `wgan-gp/wgan-gp_135k.ipynb`                       | Train WGAN-GP on 135k images |
| `esrgan_imitation_v1/esrgan_imitation_v1_2x.ipynb`| Train ESRGAN V1 (2×) |
| `esrgan_imitation_v1/esrgan_imitation_v1_4x.ipynb`| Train ESRGAN V1 (4×) |
| `esrgan_imitation_v2/esrgan_imitation_v2_4x.ipynb`| Train ESRGAN V2 (4×) |
| `real-esrgan/setup-Real-ESRGAN-Anime.ipynb`        | Load pretrained Real-ESRGAN Anime model |

---

## 📊 Experiments & Visualizations

The folder `experiment_images/` includes:

- GAN outputs over training epochs  
- Super-resolution comparisons across models  
- Final ablation study results with/without upscaling  
- Loss curves for DCGAN and WGAN-GP training  

---

## 📎 Datasets

We used two public Kaggle datasets:

- [soumikrakshit/anime-faces (~40k)](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
- [ziggykor/anime-face-dataset-expanded-2024 (~135k)](https://www.kaggle.com/datasets/ziggykor/anime-face-dataset-expanded-2024)

---



### 🔧 How to Load the Models

All 7 trained models are automatically loadable using the same architecture defined in our codebase. Use the following code snippet to get started:

```python
# 📦 Import dependencies
import torch
from model_defs import ImageGenerator, ESRGANGenerator  # your architecture definitions

# Select device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
print("Using device:", device)

# Load pretrained weights
dataset_name_40k = "soumikrakshit_anime-faces"
dataset_name_135k = "ziggykor_anime-face-dataset-expanded-2024"

# Load DCGAN and WGAN-GP models
checkpoint = torch.load(f'dcgan/models/{dataset_name_40k}/model_checkpoint_latest.pth', map_location=device)
dcgan_generator_40k = ImageGenerator().to(device)
dcgan_generator_40k.load_state_dict(checkpoint['generator_state_dict'])
dcgan_generator_40k.eval()

# Repeat similarly for:
# - dcgan_generator_135k (epoch 30)
# - wgan_gp_generator_40k
# - wgan_gp_generator_135k

# Load ESRGAN super-resolution models
checkpoint = torch.load('esrgan_imitation_v2/models/final_generator_4x.pth', map_location=device)
esrgan_v2_4x = ESRGANGenerator(scale=4).to(device)
esrgan_v2_4x.load_state_dict(checkpoint['generator_state_dict'])
esrgan_v2_4x.eval()

# Repeat for esrgan_v1_2x, esrgan_v1_4x
```

---

### 🧪 Generate and Enhance Images

You can generate synthetic anime faces and upscale them using:

```python
def get_fixed_latents(count, seed=42):
    torch.manual_seed(seed)
    return torch.randn(count, 100, 1, 1, device=device)

latents = get_fixed_latents(4)

# Generate 64x64 images using DCGAN-40k
with torch.no_grad():
    images = dcgan_generator_40k(latents)
    images = (images + 1) / 2

# Upscale to 256x256 using ESRGAN V2
with torch.no_grad():
    upscaled = esrgan_v2_4x(images).clamp(0, 1)

# Visualize
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

plt.imshow(make_grid(upscaled, nrow=2).permute(1, 2, 0).cpu())
plt.axis("off")
plt.title("Generated + Upscaled Anime Faces")
plt.show()
```

---



## 📌 Citation

If you build on this pipeline or models, please consider citing:

```bibtex
@misc{animefaceganpipeline2025,
  title={Anime Face GAN Pipeline},
  author={S. Duman, Y. Berber and Project Team},
  year={2025},
  url={https://github.com/yberber/anime-face-gan-pipeline}
}
```

---

## 🧑‍💻 Author

Built by **Selin Duman** and **Yusuf Berber** as part of the course *Computer Vision* at Heidelberg University, 2025.

---