"""
OmniColor Inference Script

Lineart-guided image colorization with optional conditioning inputs:
  - lineart (required): line art / sketch image
  - prompt: text description of the desired output
  - history_frames: previously generated frames for temporal consistency
  - color_points: spatial color hints (list of dicts with center_x_ratio, center_y_ratio, color)
  - id_image: character reference image for identity preservation

Supports two model variants:
  - Without AS-Gate (v3.3): requires only `checkpoint_path`
  - With AS-Gate (v3.8): requires both `as_gate_path` and `checkpoint_path`
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image


class OmniColor:
    """Lineart-guided image colorization pipeline."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen-Image-Edit-2509",
        checkpoint_path: Optional[str] = None,
        as_gate_path: Optional[str] = None,
        use_as_gate: bool = False,
        device_id: int = 0,
        offload: bool = True,
    ):
        """
        Args:
            model_path: HuggingFace model id or local path for the base Qwen-Image-Edit model.
            checkpoint_path: Path to the fine-tuned transformer checkpoint (.pt).
            as_gate_path: (AS-Gate only) Path to the base transformer checkpoint
                                  loaded before the AS-Gate checkpoint.
            use_as_gate: If True, load the AS-Gate variant (v3.8). Otherwise load v3.3.
            device_id: CUDA device index for model placement.
            offload: If True, use sequential CPU offload to save GPU memory.
        """
        if use_as_gate:
            from pipelines.diffusers_pipeline_qwenimage_editplus_omnicolor import QwenImageEditPlusPipeline
            from pipelines.transformer_qwenimage_priority_omnicolor import QwenImageTransformer2DModel

            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                ignore_mismatched_sizes=True,
                transformer=QwenImageTransformer2DModel.from_pretrained(
                    model_path,
                    subfolder="transformer",
                    torch_dtype=torch.bfloat16,
                ),
            )

            # Stage 1: load base transformer weights
            if as_gate_path is None:
                raise ValueError("as_gate_path is required when use_as_gate=True")
            ckpt = torch.load(as_gate_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.pipe.transformer.load_state_dict(ckpt, strict=False, assign=True)
            if missing:
                logging.warning(f"Missing keys (base checkpoint): {missing}")
            logging.info(f"Loaded base checkpoint: {as_gate_path}")

            # Stage 2: load AS-Gate transformer weights
            if checkpoint_path is None:
                raise ValueError("checkpoint_path is required when use_as_gate=True")
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.pipe.transformer.load_state_dict(ckpt, strict=False, assign=True)
            if missing:
                logging.warning(f"Missing keys (AS-Gate checkpoint): {missing}")
            logging.info(f"Loaded AS-Gate checkpoint: {checkpoint_path}")

        else:
            from pipelines.diffusers_pipeline_qwenimage_editplus_wo_as_gate import QwenImageEditPlusPipeline

            self.pipe = QwenImageEditPlusPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )

            if checkpoint_path is not None:
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                self.pipe.transformer.load_state_dict(ckpt, strict=False)
                logging.info(f"Loaded checkpoint: {checkpoint_path}")

        if offload:
            self.pipe.enable_sequential_cpu_offload(gpu_id=device_id)
        else:
            self.pipe.to(f"cuda:{device_id}")

        logging.info("OmniColor model ready.")

    # ------------------------------------------------------------------
    # Color-point overlay
    # ------------------------------------------------------------------
    @staticmethod
    def add_color_points_to_image(
        image: Image.Image,
        color_points: List[Dict],
        patch_size: int = 20,
    ) -> Image.Image:
        """Draw color hint patches onto a lineart image.

        Each element in *color_points* should be a dict::

            {
                "center_x_ratio": float,   # 0-1, relative x position
                "center_y_ratio": float,   # 0-1, relative y position
                "color": {"r": int, "g": int, "b": int}
            }
        """
        if not color_points:
            return image

        img_array = np.array(image)
        img_width, img_height = image.size
        half = patch_size // 2

        for cp in color_points:
            cx = int(cp["center_x_ratio"] * img_width)
            cy = int(cp["center_y_ratio"] * img_height)
            color = np.array(
                [cp["color"]["r"], cp["color"]["g"], cp["color"]["b"]],
                dtype=np.uint8,
            )
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(img_width, cx + half), min(img_height, cy + half)
            img_array[y1:y2, x1:x2] = color

        return Image.fromarray(img_array, "RGB")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        lineart: Image.Image,
        prompt: Optional[str] = None,
        history_frames: Optional[List[Image.Image]] = None,
        color_points: Optional[List[Dict]] = None,
        id_image: Optional[Image.Image] = None,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Generate a colorized image from lineart with optional conditioning.

        Args:
            lineart: Line art / sketch image (required).
            prompt: Text description of the scene. If None, a default prompt is used.
            history_frames: List of previously generated frames (most recent first)
                            for temporal consistency. Can be one or multiple frames.
            color_points: Spatial color hints to overlay on the lineart.
            id_image: Character reference image for identity preservation.
            num_inference_steps: Number of denoising steps.
            true_cfg_scale: Classifier-free guidance scale.
            seed: Random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        # --- Build condition images & types ---
        cond_images: List[Image.Image] = []
        cond_types: List[str] = []

        # 1. Lineart (required, always first)
        lineart_img = lineart.convert("RGB")
        if color_points:
            lineart_img = self.add_color_points_to_image(lineart_img, color_points)
        cond_images.append(lineart_img)
        cond_types.append("sketch")

        # 2. History frames (optional)
        if history_frames:
            # Most recent frame is treated as "latest_frame"
            cond_images.append(history_frames[0].convert("RGB"))
            cond_types.append("latest_frame")
            # Additional frames as "other_frames"
            for frame in history_frames[1:]:
                cond_images.append(frame.convert("RGB"))
                cond_types.append("other_frames")

        # 3. ID / character reference (optional)
        if id_image is not None:
            cond_images.append(id_image.convert("RGB"))
            cond_types.append("character")

        # --- Build prompt ---
        parts = []
        if history_frames:
            parts.append(
                "Given the current lineart and the latest frame, generate a high-quality anime frame "
                "that follows the lineart exactly and ensures consistency in coloring, background, "
                "and style with the latest frame."
            )
            if len(history_frames) > 1:
                parts.append(
                    "Use other frames as general references to maintain overall consistency."
                )
        else:
            parts.append(
                "Given the current lineart, generate a high-quality anime frame "
                "that follows the lineart exactly."
            )

        if color_points:
            parts.append(
                "The lineart contains colored grid cells indicating the desired colors for different regions."
            )

        if id_image is not None:
            parts.append(
                "Use the character reference image to ensure consistency in character appearance."
            )

        if prompt:
            parts.append(f"Frame Description: {prompt}")

        full_prompt = " ".join(parts)

        # --- Generate ---
        generator = torch.manual_seed(seed) if seed is not None else None

        with torch.inference_mode():
            result = self.pipe(
                image=cond_images,
                prompt=full_prompt,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=" ",
                num_inference_steps=num_inference_steps,
                cond_types=cond_types,
                generator=generator,
            )

        return result.images[0]


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="OmniColor Inference")
    parser.add_argument("--lineart", type=str, required=True, help="Path to input lineart image")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt describing the scene")
    parser.add_argument("--history_frames", type=str, nargs="*", default=None,
                        help="Paths to history frame images (most recent first)")
    parser.add_argument("--color_points", type=str, default=None,
                        help="Path to a JSON file containing color point hints")
    parser.add_argument("--id_image", type=str, default=None, help="Path to character reference image")
    parser.add_argument("--output", type=str, default="./example/output.png", help="Output image path")

    # Model configuration
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen-Image-Edit-2509",
                        help="HuggingFace model id or local path for base model")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to fine-tuned transformer checkpoint (.pt)")
    parser.add_argument("--as_gate_path", type=str, default=None,
                        help="AS-Gate Path to AS-Gate checkpoint")
    parser.add_argument("--use_as_gate", action="store_true",
                        help="Use AS-Gate")
    parser.add_argument("--device_id", type=int, default=0, help="CUDA device index")
    parser.add_argument("--no_offload", action="store_true",
                        help="Disable sequential CPU offload (requires more GPU memory)")

    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    # --- Load model ---
    model = OmniColor(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        as_gate_path=args.as_gate_path,
        use_as_gate=args.use_as_gate,
        device_id=args.device_id,
        offload=not args.no_offload,
    )

    # --- Prepare inputs ---
    lineart = Image.open(args.lineart).convert("RGB")

    history_frames = None
    if args.history_frames:
        history_frames = [Image.open(p).convert("RGB") for p in args.history_frames]

    color_points = None
    if args.color_points:
        import json
        with open(args.color_points, "r") as f:
            cp_data = json.load(f)
        # Support dict format {lineart_path: [points...]} or plain list format [points...]
        if isinstance(cp_data, dict):
            # Look up by lineart path (try exact match, then basename)
            color_points = cp_data.get(args.lineart)
            if color_points is None:
                lineart_basename = os.path.basename(args.lineart)
                for key, val in cp_data.items():
                    if os.path.basename(key) == lineart_basename:
                        color_points = val
                        break
            if color_points is None:
                logging.warning(f"No color points found for lineart: {args.lineart}")
        else:
            color_points = cp_data

    id_image = None
    if args.id_image:
        id_image = Image.open(args.id_image).convert("RGB")

    # --- Generate ---
    result = model.generate(
        lineart=lineart,
        prompt=args.prompt,
        history_frames=history_frames,
        color_points=color_points,
        id_image=id_image,
        num_inference_steps=args.num_inference_steps,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
    )

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.save(args.output)
    logging.info(f"Saved result to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )
    main()
