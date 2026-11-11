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

    def __init__(self, corruption=None, severity=1, **kwargs):
        super().__init__(**kwargs)

        self.corruption = corruption
        self.severity = severity
        self.rng = np.random.default_rng(42)

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