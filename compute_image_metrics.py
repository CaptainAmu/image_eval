#!/usr/bin/env python3
"""
Compute CLIP Score and LAION Aesthetic Score for generated images
against their corresponding prompts (prompt_c column in CSV).

Usage:
  pip install -r requirements_image_metrics.txt
  python compute_image_metrics.py --images_dir /path/to/images --csv_path data.csv --output results.csv

CSV must have:
  - A column with image filenames (e.g. "12.png" or "12") -- set via --image_column if not auto-detected
  - A column "prompt_c" (or --prompt_column) with the text prompt for each image

Example CSV:
  image,log_weights,prompt_c
  12.png,-0.5,"a cat sitting on a mat"
  18.png,-0.3,"a dog in the park"

Output: same CSV with added columns clip_score, laion_score (depending on --metrics).
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


def get_clip_score_model(device):
    """Load CLIP model for image-text similarity (CLIP Score)."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def compute_clip_scores(model, processor, image_paths, prompts, device, batch_size=8):
    """Compute CLIP Score (cosine similarity * 100) for each image-prompt pair."""
    from torch.nn.functional import cosine_similarity

    scores = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP Score"):
            batch_paths = image_paths[i : i + batch_size]
            batch_prompts = prompts[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(
                text=batch_prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            outputs = model(**inputs)
            # Cosine similarity between image and text embeddings (normalized)
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            sim = (image_emb * text_emb).sum(dim=-1)
            # CLIP Score formula: max(100 * cos_sim, 0)
            batch_scores = (100.0 * sim.clamp(min=0)).cpu().tolist()
            scores.extend(batch_scores)
    return scores


def get_laion_model(device):
    """Load CLIP + LAION aesthetic linear head for aesthetic score."""
    from urllib.request import urlretrieve

    from transformers import CLIPModel, CLIPProcessor

    home = Path.home()
    cache_folder = home / ".cache" / "laion_aesthetic"
    cache_folder.mkdir(parents=True, exist_ok=True)
    path_to_model = cache_folder / "sa_0_4_vit_l_14_linear.pth"
    if not path_to_model.exists():
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/"
            "sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    linear = torch.nn.Linear(768, 1).to(device)
    linear.load_state_dict(torch.load(path_to_model, map_location=device))
    linear.eval()
    return clip_model, processor, linear


def compute_laion_scores(clip_model, processor, linear, image_paths, device, batch_size=8):
    """Compute LAION aesthetic score (0–10 scale) for each image."""
    scores = []
    clip_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="LAION"):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            out = clip_model.get_image_features(inputs["pixel_values"])
            # Newer transformers may return BaseModelOutputWithPooling; extract tensor
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                image_emb = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                image_emb = out.last_hidden_state[:, 0, :]
            else:
                image_emb = out
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            batch_scores = linear(image_emb).squeeze(-1).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
    return scores


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP Score and LAION Aesthetic for images vs prompts."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing images (e.g. 12.png, 18.png).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV with image filenames and prompt_c column.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics_results.csv",
        help="Output CSV path with added columns: clip_score, laion_score.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default=None,
        help="CSV column name for image filename. Auto-detected if not set (first column or 'image'/'filename').",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt_c",
        help="CSV column name for the text prompt (default: prompt_c).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["clip", "laion"],
        choices=["clip", "laion"],
        help="Which metrics to compute (default: clip, laion).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for CLIP/LAION (default: 8).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, cpu, or cuda:0 (default: auto).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    df = pd.read_csv(args.csv_path)
    if args.prompt_column not in df.columns:
        raise ValueError(
            f"CSV must have column '{args.prompt_column}'. Columns: {list(df.columns)}"
        )

    # Resolve image column
    if args.image_column is not None:
        if args.image_column not in df.columns:
            raise ValueError(
                f"Image column '{args.image_column}' not in CSV. Columns: {list(df.columns)}"
            )
        image_col = args.image_column
    else:
        for cand in ["image", "filename", "name", "file", "img"]:
            if cand in df.columns:
                image_col = cand
                break
        else:
            image_col = df.columns[0]
            print(f"Using first column as image names: '{image_col}'")

    # Build full paths and filter to existing files
    image_paths = []
    valid_indices = []
    for idx, row in df.iterrows():
        fname = str(row[image_col]).strip()
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            fname_candidates = [fname + ".png", fname + ".jpg", fname + ".jpeg"]
        else:
            fname_candidates = [fname]
        path = None
        for fc in fname_candidates:
            p = images_dir / fc
            if p.exists():
                path = p
                break
        if path is not None:
            image_paths.append(path)
            valid_indices.append(idx)
        else:
            print(f"Warning: image not found {images_dir / fname}")

    if not valid_indices:
        raise FileNotFoundError(
            f"No images found in {images_dir} matching CSV column '{image_col}'."
        )

    df_sub = df.loc[valid_indices].copy()
    prompts = df_sub[args.prompt_column].astype(str).tolist()

    if "clip" in args.metrics:
        print("Loading CLIP model...")
        clip_model, clip_processor = get_clip_score_model(device)
        df_sub["clip_score"] = compute_clip_scores(
            clip_model, clip_processor, image_paths, prompts, device, args.batch_size
        )
        del clip_model, clip_processor
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    if "laion" in args.metrics:
        print("Loading LAION aesthetic model...")
        clip_model, processor, linear = get_laion_model(device)
        df_sub["laion_score"] = compute_laion_scores(
            clip_model, processor, linear, image_paths, device, args.batch_size
        )
        del clip_model, processor, linear
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    df_sub.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")
    score_cols = [c for c in ["clip_score", "laion_score"] if c in df_sub.columns]
    if score_cols:
        print(df_sub[score_cols].describe())
    print(df_sub.head())


if __name__ == "__main__":
    main()
