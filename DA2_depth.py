import os
import numpy as np
import cv2
from transformers import pipeline
import torch


def run_depth_anything_v2(img_paths, output_path, encoder: str = "vits", input_size: int = 518):
    # Map to HF pipeline model ids
    model_id = {
        'vits': 'depth-anything/Depth-Anything-V2-Small-hf',
        'vitb': 'depth-anything/Depth-Anything-V2-Base-hf',
        'vitl': 'depth-anything/Depth-Anything-V2-Large-hf',
    }.get(encoder, 'depth-anything/Depth-Anything-V2-Small-hf')

    # Choose device
    device = (
        0 if torch.cuda.is_available() else
        "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
        -1
    )
    pipe = pipeline(task="depth-estimation", model=model_id, device=device)

    os.makedirs(output_path, exist_ok=True)
    for img_path in img_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        # Let pipeline load from path (formats it expects) to avoid type issues
        out = pipe(img_path)
        depth_img = out["depth"]  # PIL Image
        depth = np.array(depth_img).astype(np.float32)
        base = os.path.splitext(os.path.basename(img_path))[0]
        np.save(os.path.join(output_path, base + '.npy'), depth)
        # Save 16-bit png for preview
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax - dmin > 1e-8:
            d16 = (65535.0 * (depth - dmin) / (dmax - dmin)).astype(np.uint16)
        else:
            d16 = np.zeros_like(depth, dtype=np.uint16)
        cv2.imwrite(os.path.join(output_path, base + '.png'), d16)


