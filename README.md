```markdown
# Hierarchical Part-Prototype Learning for Fine-Grained Wildlife-Image Classification (HiPP-Learning)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20092941.svg)](https://doi.org/10.5281/zenodo.20092941)

This repository contains the official PyTorch implementation for the manuscript "**Hierarchical Part-Prototype Learning for Fine-Grained Wildlife-Image Classification**", submitted to the *Journal of Electronic Imaging*.
## 📝 Introduction
Fine-grained wildlife classification (FGWC) is challenging because visually similar species must be distinguished under complex real-world conditions, such as occlusion, camouflage, blur, and cluttered backgrounds. 

To address this problem using image-level labels only, we propose **HiPP-Learning**, a unified framework that learns discriminative part prototypes across multiple semantic stages and progressively aligns them with global object representations. The framework features two core modules:
* **Soft Part-Prototype Mining (SPPM):** Discovers semantically meaningful local regions and yields robust part representations while reducing background distraction.
* **Part-Channel Interactive Attention (PCIA):** Enhances representation learning by jointly modeling part importance and channel responses for adaptive local-global feature fusion.

## ⚙️ Environment Setup
The code has been tested with Python 3.8+ and PyTorch. We recommend using Conda to set up the environment.

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate hipp_learning

```

## 📂 Data Preparation

We evaluate our model on four fine-grained wildlife datasets: **NXbirds**, **Butterfly200**, **Snake135**, and **WildFish**.

Please download the datasets and organize them using the standard `ImageFolder` structure. The directory layout should look like this:

```text
datasets/
├── butterfly200/
│   ├── train/
│   │   ├── class_001/
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   ├── class_002/
│   │   └── ...
│   ├── val/
│   │   ├── class_001/
│   │   ├── class_002/
│   │   └── ...
├── nxbirds/
├── wildfish/
└── snake135/

```

*(Note: To comply with the original dataset licenses, we do not distribute the raw image files directly. Please download the raw datasets from their official release repositories (and Kaggle for Snake135). You can then use the exact train/test split files provided in this repository to reorganize them. 链接: https://pan.baidu.com/s/1c99puD7Yn52Ubu0TS82k8A?pwd=d4x6 提取码: d4x6)*
* ## 🚀 Training & Evaluation
We provide straightforward commands to reproduce the experiments.

### 1. Train from scratch

To train the HiPP-Learning model on the Butterfly200 dataset (or any other dataset by changing the `--dataset` argument), run:

```bash
python main.py \
    --model_name HiPP_butterfly200 \
    --data_root ./datasets \
    --dataset butterfly200 \
    --num_parts 4 \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4

```

Checkpoints and tensorboard logs will be saved in `./<dataset_name>/<model_name>/` and `./results_<model_name>/` respectively.

### 2. Evaluation / Testing

To evaluate a pre-trained model and generate visual attention maps (e.g., Grad-CAM / Prototype visualizations), use the `--only_test` flag and provide the checkpoint path:

```bash
python main.py \
    --model_name HiPP_butterfly200_Eval \
    --dataset butterfly200 \
    --only_test \
    --pretrained_model_path ./butterfly200/HiPP_butterfly200_best.pt \
    --save_figures

```

## 📦 Pre-trained Models

You can download our pre-trained model weights for the four datasets from the following links to reproduce the results reported in the paper:

* [NXbirds Checkpoint](https://www.google.com/search?q=https://pan.baidu.com/s/1FPmpwqjH0WtCpCBw-3uZUw) (提取码: `y9x4`)
* [Butterfly200 Checkpoint](https://www.google.com/search?q=https://pan.baidu.com/s/1nh7lQg_zN1V-V0R826DbMw) (提取码: `qqdm`)
* [Snake135 Checkpoint]https://pan.baidu.com/s/1dJ6ZUlNnpVNYvN7qsnIJqw?pwd=my7v 提取码: my7v
* [WildFish Checkpoint](https://www.google.com/search?q=https://pan.baidu.com/s/1ftHNQPE2gr8F0ciwL9CfcA) (提取码: `rfvf`)

## 🎓 Citation

If you find our code, datasets, or this research helpful for your work, please consider citing our paper:

```bibtex
@article{tang2026hipplearning,
  title={Hierarchical part-prototype learning for fine-grained wildlife-image classification},
  author={Tai, Xiao and Tang, Jun and Wang, Bin},
  journal={Journal of Electronic Imaging},
  year={2026},
  note={Under review}
}
## ✉️ Contact

For any questions regarding the code or the paper, please open an issue or contact the corresponding author at `wangbin@nufe.edu.cn`.

```

```
