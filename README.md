# OmniColor: A Unified Framework for Multi-modal Lineart Colorization

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

## 🚀 Usage

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

### Step 2: Create a Conda Environment
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
