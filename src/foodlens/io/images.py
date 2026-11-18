"""Load raw frames as RGB uint8 arrays at a working resolution (never writes to the raw data).

JPEG frames are decoded with Pillow's ``draft`` mode (DCT down-scaling), which is several times
faster than a full 16 MP decode, then resized so the long side equals ``long_side`` px.
``ImageOps.exif_transpose`` is always applied.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from foodlens.errors import E_DECODE, DataError


def load_rgb(path: str | Path, long_side: int | None = None) -> np.ndarray:
    """RGB uint8 array (H, W, 3); long side scaled down to ``long_side`` px if given.

    Raises ``DataError(E_DECODE)`` if the file cannot be decoded.
    """
    try:
        with Image.open(path) as im:
            if long_side is not None and max(im.size) > long_side:
                s = long_side / max(im.size)
                im.draft("RGB", (round(im.size[0] * s), round(im.size[1] * s)))
            img = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise DataError(E_DECODE, f"{Path(path).name}: {exc}") from exc
    if long_side is not None and max(img.size) > long_side:
        img.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)
