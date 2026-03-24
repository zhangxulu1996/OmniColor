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


## Features

- **Lineart** (required): lineart image as structural guidance
- **Text prompt** (optional): scene description to guide colorization
- **History frames** (optional): one or more previously generated frames for temporal consistency
- **Color points** (optional): spatial color hints overlaid on the lineart
- **ID image** (optional): character reference image for identity preservation

Two model variants are supported:
| Variant | Flag | Checkpoints needed |
|---------|------|--------------------|
| Without AS-Gate | *(default)* | `--checkpoint_path` |
| With AS-Gate | `--use_as_gate` | `--as_gate_path` + `--checkpoint_path` |

## Usage

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

### Step 3: Run without AS-Gate

```bash
python inference.py \
    --lineart example/inputs/0.png \
    --prompt "A blonde-haired young man with green eyes, wearing a dark jacket." \
    --checkpoint_path checkpoints/model.pt \
    --output example/output.png
```

### Run With AS-Gate + all conditions

```bash
python inference.py \
    --lineart example/inputs/0.png \
    --prompt "A blonde-haired young man with green eyes, wearing a dark jacket." \  # optional
    --color_points example/color_points.json \  # optional
    --history_frames example/output_frame_01.png \  # optional
    --id_image example/id_ref.png \  # optional
    --use_as_gate \
    --as_gate_path checkpoints/as_gate.pt \
    --checkpoint_path checkpoints/model.pt \
    --output example/output.png
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
