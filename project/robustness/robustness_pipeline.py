from mmdet3d.datasets.pipelines.loading import PrepareImageInputs
from mmdet3d.datasets.builder import PIPELINES
import numpy as np

from project.robustness.image_corruption import (
    add_gaussian_noise,
    add_gaussian_blur,
)


@PIPELINES.register_module()
class RobustPrepareImageInputs(PrepareImageInputs):
    """Image preprocessing with optional robustness testing."""

    def __init__(self, corruption=None, severity=1, crop_shift=0, **kwargs):
        super().__init__(**kwargs)

        self.corruption = corruption
        self.severity = severity
        self.crop_shift = crop_shift
        self.rng = np.random.default_rng(42)

    def sample_augmentation(self, H, W, flip=None, scale=None):
        resize, resize_dims, crop, flip, rotate = super().sample_augmentation(
            H, W, flip=flip, scale=scale
        )
        if not self.is_train and self.crop_shift > 0:
            x1, y1, x2, y2 = crop
            new_y1 = max(0, y1 - self.crop_shift)
            crop = (x1, new_y1, x2, new_y1 + (y2 - y1))
        return resize, resize_dims, crop, flip, rotate

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # First, apply ALOcc's original image transformations
        img = super().img_transform_core(
            img, resize_dims, crop, flip, rotate
        )

        # Then apply our selected corruption
        if self.corruption == "gaussian_noise":
            img = add_gaussian_noise(
                img,
                sigma=15 * self.severity,
                # seed=int(self.rng.integers(0, 2**32 - 1))
                seed=42
            )

        elif self.corruption == "gaussian_blur":
            img = add_gaussian_blur(
                img, radius=2 * self.severity
            )

        return img