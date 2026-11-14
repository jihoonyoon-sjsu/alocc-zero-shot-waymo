"""Image corruption utilities for ALOcc Waymo robustness experiments."""

import numpy as np
from PIL import Image, ImageFilter


def add_gaussian_noise(image, sigma=15, seed=42):
    """Add Gaussian noise to a camera image."""

    rng = np.random.default_rng(seed)

    # Convert image pixels into numbers
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)

    # Generate random noise
    noise = rng.normal(0, sigma, size=pixels.shape)

    # Add noise and keep pixel values between 0 and 255
    noisy_pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy_pixels)

def add_gaussian_blur(image, radius=2):
    """Apply Gaussian blur to a camera image."""
    return image.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=radius)
    )