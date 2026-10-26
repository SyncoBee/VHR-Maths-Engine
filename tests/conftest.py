import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def photo_4x3(tmp_path_factory):
    """Synthetic 4:3 'photo': gradients + soft blobs + saturated areas."""
    H, W = 240, 320
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3))
    img[..., 0] = xx / W
    img[..., 1] = yy / H
    img[..., 2] = 0.5 + 0.5 * np.sin(xx / 20) * np.cos(yy / 15)
    for cx, cy, r, c in [(80, 60, 20, (1, 1, 0)), (240, 150, 35, (0, 1, 1)), (150, 200, 12, (1, 0, 1))]:
        d = np.hypot(xx - cx, yy - cy)
        img += np.exp(-(d / r) ** 4)[..., None] * np.array(c)
    arr = (np.clip(img * 1.25 - 0.1, 0, 1) * 255).astype(np.uint8)       # pushes parts into 0 / 255 saturation
    path = tmp_path_factory.mktemp("img") / "photo.png"
    Image.fromarray(arr).save(path)
    return str(path), arr
