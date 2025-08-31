import cv2
import numpy as np
import torch
import os
import shutil
from PIL import Image
from collections import namedtuple
import math


CACHE_DIR = '.cache'

def clear_cache():
    """Clears the cache directory."""
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

def save_image_to_cache(image: np.ndarray, filename: str):
    """Saves a BGR numpy image to the cache as an RGB image."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    save_path = os.path.join(CACHE_DIR, filename)
    pil_img.save(save_path)


def bgr_image_to_rgb_tensor(img: np.ndarray) -> torch.Tensor:
    """Convert a BGR NumPy image in [0..1] range to a PyTorch RGB float32 tensor."""
    assert img.shape[2] == 3, "image must be RGB"
    if img.dtype == "float64":
        img = img.astype("float32")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()


def rgb_tensor_to_bgr_image(tensor: torch.Tensor, *, min_max=(0.0, 1.0)) -> np.ndarray:
    """
    Convert a PyTorch RGB tensor in range `min_max` to a BGR NumPy image in [0..1] range.
    """
    tensor = tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])
    assert tensor.dim() == 3, "tensor must be RGB"
    img_np = tensor.numpy().transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image, no RGB/BGR required
        return np.squeeze(img_np, axis=2)
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)


class Grid(namedtuple("_Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])):
    @property
    def tile_count(self) -> int:
        """
        The total number of tiles in the grid.
        """
        return sum(len(row[2]) for row in self.tiles)


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w, h = image.size

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image