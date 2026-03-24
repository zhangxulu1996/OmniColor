# OmniColor: A Unified Framework for Multi-modal Lineart Colorization
[![ECCV 2026 Submission](https://img.shields.io/badge/ECCV-2026%20Submission-4b4b96.svg)](https://eccv2026.ecva.net/)
[![License](https://img.shields.io/github/license/your-username/OmniColor.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

Official open-source implementation of **OmniColor**, a unified framework for multi-modal lineart colorization that supports arbitrary combinations of control signals, submitted to ECCV 2026 (Paper ID: 5503).

OmniColor achieves superior controllability, visual quality, and temporal stability for lineart colorization tasks, and provides a robust practical solution for animation production, comic creation, game design and other professional content creation scenarios.

## 📝 Paper Abstract
Lineart colorization is a critical stage in professional content creation, yet achieving precise and flexible results under diverse user constraints remains a significant challenge. To address this, we propose OmniColor, a unified framework for multi-modal lineart colorization that supports arbitrary combinations of control signals. Specifically, we systematically categorize guidance signals into two types: spatially-aligned conditions and semantic-reference conditions. For spatially-aligned inputs, we employ a dual-path encoding strategy paired with a Dense Feature Alignment loss to ensure rigorous boundary preservation and precise color restoration. For semantic-reference inputs, we utilize a VLM-only encoding scheme integrated with a Temporal Redundancy Elimination mechanism to filter repetitive information and enhance inference efficiency. To resolve potential input conflicts, we introduce an Adaptive Spatial-Semantic Gating module that dynamically balances multi-modal constraints. Experimental results demonstrate that OmniColor achieves superior controllability, visual quality, and temporal stability, providing a robust and practical solution for lineart colorization.

## 🌟 Key Features
- **Unified Multi-modal Control**: Supports arbitrary combinations of lineart, text prompts, color hints, identity references, temporal history frames and other control signals
- **Two-category Condition Encoding**: Spatially-aligned (pixel-level constraint) and semantic-reference (high-level guidance) condition separation for targeted processing
- **Dual-path Encoding for Spatial Conditions**: VAE + VLM dual encoder with Dense Feature Alignment (DFA) loss for precise boundary preservation and color restoration
- **Efficient Semantic Encoding**: VLM-only encoding + Temporal Redundancy Elimination (TRE) mechanism for low computational cost and high inference efficiency
- **Adaptive Conflict Resolution**: Adaptive Spatial-Semantic Gating (AS-Gate) module dynamically balances multi-modal constraints and resolves input conflicts
- **High Temporal Stability**: Excellent consistency for sequential/ video lineart colorization, suitable for animation production
- **Professional Grade Results**: Achieves state-of-the-art performance in both quantitative metrics and human user studies

## 🛠️ Installation
### Prerequisites
- Python 3.10 or higher
- PyTorch 2.0 or higher
- CUDA 11.7 or higher (for GPU acceleration)
- 8+ NVIDIA GPUs (recommended for training, single GPU for inference)

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/OmniColor.git
cd OmniColor
```

### Step 2: Create a Conda Environment (Optional)
```bash
conda create -n omnicolor python=3.10
conda activate omnicolor
```

### Step 3: Install Dependencies
```bash
# Install PyTorch (adjust according to your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Weights
Download the pre-trained base model weights and place them in the `pretrained/` directory:
- Qwen2.5-VL-7B: [Official Link](https://modelscope.cn/models/qwen/Qwen2.5-VL-7B/summary)
- Qwen-Image-Edit-2509: [Official Link](https://modelscope.cn/models/qwen/Qwen-Image-Edit/summary)
- DINOv3: [Official Link](https://github.com/facebookresearch/dinov3)

```bash
mkdir -p pretrained
# Place downloaded weights in pretrained/
```

## 📚 Dataset Preparation
### Dataset Overview
OmniColor is trained on a large-scale lineart colorization dataset curated from 25 high-quality animation series, containing **120,000 high-quality pairs** with six types of auxiliary signals. The test set is constructed from 6 unseen animation series with **900 test samples** and 305 shots.

### Dataset Construction Pipeline
1. **Raw Data Collection**: Extract frames (≥1280×720) from animation series at 1 FPS
2. **Shot Boundary Detection**: Use [pySceneDetect](https://github.com/Breakthrough/PySceneDetect) to define temporal relationships
3. **Spatially-aligned Condition Extraction**:
   - Lineart: GAN-based framework optimized for anime style
   - Recent history frames: Immediate preceding frame in the same shot
   - Color hints: Random uniform pixel blocks (≥10×10) sampled from ground truth
4. **Semantic-reference Condition Extraction**:
   - Text descriptions: Hierarchical captions (shot-level + frame-level) generated by Qwen3-VL-32B
   - ID reference maps: SAM3 for segmentation + CLIP for cross-frame identity matching
   - Long-term history frames: Sampled from adjacent distinct shots (processed by TRE)

### Dataset Structure
Organize your dataset in the following structure:
```
dataset/
├── train/
│   ├── lineart/
│   ├── gt/
│   ├── color_hints/
│   ├── text/
│   ├── id_maps/
│   ├── recent_history/
│   └── long_term_history/
└── test/
    ├── lineart/
    ├── gt/
    ├── color_hints/
    ├── text/
    ├── id_maps/
    ├── recent_history/
    └── long_term_history/
```

### Preprocessing Script
Run the provided preprocessing script to generate the required conditions from raw animation frames:
```bash
python scripts/preprocess_dataset.py --raw_data path/to/raw_frames --output path/to/dataset
```

## 🚀 Quick Start
### Inference
#### Single Image Colorization (L+T)
Lineart + Text prompt colorization (the most basic configuration):
```bash
python infer.py \
  --lineart path/to/lineart.png \
  --text "A young girl with long brown hair wearing a yellow cardigan, white shirt, and dark skirt" \
  --output path/to/result.png \
  --ckpt path/to/omnicolor_ckpt.pth
```

#### Multi-modal Control (L+T+C+I)
Lineart + Text + Color hints + Identity reference (full multi-modal configuration):
```bash
python infer.py \
  --lineart path/to/lineart.png \
  --text "A blonde-haired witch with green eyes, wearing a wide-brimmed hat" \
  --color_hints path/to/color_hints.png \
  --id_ref path/to/id_reference.png \
  --output path/to/result.png \
  --ckpt path/to/omnicolor_ckpt.pth
```

#### Sequential/Video Colorization
Lineart sequence + Text + Temporal history (for animation/video colorization):
```bash
python infer_sequential.py \
  --lineart_dir path/to/lineart_sequence \
  --text "A young man with red hair wearing a red uniform with gold trim" \
  --output_dir path/to/result_sequence \
  --ckpt path/to/omnicolor_ckpt.pth
```

### Training
#### Full Training
Train the complete OmniColor model on 8 NVIDIA H20 GPUs (recommended):
```bash
bash scripts/train_full.sh
```

#### Stage 1: Backbone Pretraining
Pretrain the MMDiT backbone for 12,000 iterations:
```bash
bash scripts/train_backbone.sh
```

#### Stage 2: AS-Gate Fine-tuning
Optimize the AS-Gate module with frozen backbone for 3,000 iterations:
```bash
bash scripts/train_as_gate.sh
```

### Evaluation
#### Quantitative Evaluation
Evaluate the model on the test set with all metrics (FID, SSIM, PSNR, LPIPS, Image-Align, ∆FC):
```bash
python evaluate.py \
  --test_dir path/to/test_dataset \
  --ckpt path/to/omnicolor_ckpt.pth \
  --output_dir path/to/evaluation_results
```

#### Qualitative Evaluation
Generate qualitative comparison results with SOTA methods:
```bash
python eval_qualitative.py \
  --test_samples path/to/test_samples \
  --ckpt path/to/omnicolor_ckpt.pth \
  --output_dir path/to/qualitative_results
```

## 🧠 Model Architecture
OmniColor is built on a **Multi-modal Diffusion Transformer (MMDiT)** backbone optimized via flow matching, with three core modules:
1. **Spatially-aligned Condition Encoder**: VAE + VLM dual encoder + DFA loss for pixel-level constraint processing
2. **Semantic-reference Condition Encoder**: VLM-only encoder + TRE mechanism for high-level guidance processing
3. **Adaptive Spatial-Semantic Gating (AS-Gate)**: Dynamically balances multi-modal features and resolves conflicts
4. **MMDiT Backbone**: Fuses all conditional features and performs diffusion-based colorization
5. **Loss Functions**: Flow Matching Loss (L_FM) + Dense Feature Alignment Loss (L_DFA)

<p align="center">
  <img src="assets/omnicolor_architecture.png" alt="OmniColor Architecture" width="800">
  <br>
  OmniColor Model Architecture
</p>

## 📊 Experimental Results
### Quantitative Results
OmniColor consistently outperforms state-of-the-art lineart colorization and multi-modal generation methods on all metrics (SSIM, PSNR, FID, LPIPS, Image-Align, ∆FC) under all input configurations.

Key quantitative results (L+T+H+G configuration):
- SSIM: **0.9769**
- PSNR: **0.9234**
- FID: **34.65**
- LPIPS: **27.76**
- Image-Align: **0.0770**
- ∆FC: **0.14**

### User Study Results
OmniColor achieves dominant preference rates in human user studies (10 participants with bachelor's degree or higher):
- **Structural Fidelity**: 81.0% (far exceeding the second-best 9.3%)
- **Instruction Following**: 42.1%
- **Visual Quality**: 59.9%
- **Overall Preference**: 61.0%

### Qualitative Results
Qualitative comparison results with SOTA methods are available in the `assets/` directory, including:
- Prompt-driven colorization
- Reference-based colorization
- Multi-modal control colorization
- Sequential/video colorization

## 📈 Ablation Study
All key components of OmniColor contribute significantly to the performance:
- **VLM-only Encoding**: Reduces inference time from 452s to 203s with comparable performance
- **DFA Loss**: Improves SSIM from 0.7287 to 0.7500 and recovers high-frequency structural details
- **TRE Mechanism**: Reduces history tokens by 25% (7521→5203) and boosts inference efficiency
- **AS-Gate Module**: Achieves the best performance across all quality metrics by dynamically balancing multi-modal constraints

## 📁 Project Structure
```
OmniColor/
├── assets/              # Architecture figures, qualitative results, demo images
├── configs/             # Training and inference configuration files
├── data/                # Dataset loading and preprocessing code
├── models/              # Model architecture implementation
│   ├── omnicolor.py     # Main OmniColor model
│   ├── encoders/        # Spatial and semantic encoders
│   ├── backbone/        # MMDiT backbone
│   ├── as_gate.py       # Adaptive Spatial-Semantic Gating module
│   ├── tre.py           # Temporal Redundancy Elimination mechanism
│   └── losses.py        # Loss functions (L_FM, L_DFA)
├── pretrained/          # Pre-trained base model weights
├── scripts/             # Training, evaluation and preprocessing scripts
├── infer.py             # Single image inference script
├── infer_sequential.py  # Sequential/video inference script
├── train.py             # Main training script
├── evaluate.py          # Quantitative evaluation script
├── eval_qualitative.py  # Qualitative evaluation script
├── requirements.txt     # Dependencies
├── LICENSE              # License file
└── README.md            # Project README
```

## 📜 License
This project is released under the **Apache License 2.0** license. See [LICENSE](LICENSE) for more details.

## 📖 Citation
If you find OmniColor useful in your research or work, please cite our paper:
```bibtex
@inproceedings{omnicolor2026,
  title={OmniColor: A Unified Framework for Multi-modal Lineart Colorization},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
*(Will be updated with author information after paper acceptance)*

## 🚨 Note
This is the official open-source implementation of the OmniColor paper submitted to ECCV 2026. The code and pre-trained models are released upon paper acceptance. For the current version, the code is fully functional and the pre-trained weights will be updated soon.

## 🤝 Contributing
We welcome contributions to OmniColor! If you have any ideas, bug fixes, or feature requests, please open an issue or submit a pull request. For major changes, please discuss them with us first via issues.

## 📧 Contact
If you have any questions about the project, please contact us at: [your-email@example.com]

---
**OmniColor** | ECCV 2026 Submission | A Unified Framework for Multi-modal Lineart Colorization
