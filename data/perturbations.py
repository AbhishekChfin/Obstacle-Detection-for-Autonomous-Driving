"""
Simulated adverse weather and imaging perturbations.
Implements the 9 test conditions described in Section 3.5 & evaluated in Section 4.6 (Table 2 & 3).
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def apply_gaussian_noise(pil_img: Image.Image, std: float = 0.05) -> Image.Image:
    """Add additive zero-mean Gaussian noise (std=0.05)."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, std, arr.shape)
    noisy_arr = np.clip(arr + noise, 0.0, 1.0) * 255.0
    return Image.fromarray(noisy_arr.astype(np.uint8))


def apply_salt_pepper_noise(pil_img: Image.Image, amount: float = 0.02) -> Image.Image:
    """Add salt-and-pepper impulse noise (amount=0.02)."""
    arr = np.array(pil_img).copy()
    h, w, c = arr.shape
    num_noisy = int(amount * h * w)

    # Salt (white pixels)
    num_salt = num_noisy // 2
    coords_y = np.random.randint(0, h, num_salt)
    coords_x = np.random.randint(0, w, num_salt)
    arr[coords_y, coords_x, :] = 255

    # Pepper (black pixels)
    num_pepper = num_noisy - num_salt
    coords_y = np.random.randint(0, h, num_pepper)
    coords_x = np.random.randint(0, w, num_pepper)
    arr[coords_y, coords_x, :] = 0

    return Image.fromarray(arr)


def apply_blur(pil_img: Image.Image, radius: float = 3.0) -> Image.Image:
    """Apply Gaussian blur with kernel radius=3."""
    return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_brightness(pil_img: Image.Image, factor: float) -> Image.Image:
    """Scale image brightness by a multiplier factor (e.g. 0.70 for -30%, 1.30 for +30%)."""
    enhancer = ImageEnhance.Brightness(pil_img)
    return enhancer.enhance(factor)


def apply_contrast(pil_img: Image.Image, factor: float) -> Image.Image:
    """Scale image contrast by factor (e.g. 0.70 for low contrast, 1.30 for high contrast)."""
    enhancer = ImageEnhance.Contrast(pil_img)
    return enhancer.enhance(factor)


def apply_fog(pil_img: Image.Image, opacity: float = 0.40) -> Image.Image:
    """
    Simulated fog: additive alpha-blended grayish-white fog mask (opacity=0.4).
    Simulates atmospheric scattering reducing visibility and contrast.
    """
    arr = np.array(pil_img).astype(np.float32)
    # Atmospheric fog light (typically high intensity with slight warm/cool tint)
    fog_color = np.array([210.0, 215.0, 220.0], dtype=np.float32)
    fogged = (1.0 - opacity) * arr + opacity * fog_color
    fogged = np.clip(fogged, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(fogged)


def apply_rain(pil_img: Image.Image, num_streaks: int = 400) -> Image.Image:
    """
    Simulated rain: semi-transparent diagonal falling streaks across the image.
    Following Section 3.5 & Haloi et al. [19].
    """
    arr = np.array(pil_img).copy()
    h, w, _ = arr.shape

    # Draw semi-transparent rain streaks
    length = np.random.randint(25, 45, size=num_streaks)
    angle_offset = np.random.randint(4, 9, size=num_streaks)
    xs = np.random.randint(0, w - 15, size=num_streaks)
    ys = np.random.randint(0, h - 50, size=num_streaks)

    for i in range(num_streaks):
        x0, y0 = xs[i], ys[i]
        l = length[i]
        dx = angle_offset[i]
        for step in range(l):
            curr_y = min(y0 + step, h - 1)
            curr_x = min(x0 + int(step * dx / l), w - 1)
            # Alpha blend white streak onto pixel
            arr[curr_y, curr_x] = (arr[curr_y, curr_x].astype(np.float32) * 0.4 + 255.0 * 0.6).astype(np.uint8)

    return Image.fromarray(arr)


def apply_jpeg_compression(pil_img: Image.Image, quality: int = 30) -> Image.Image:
    """Simulated compression artifacts (quality=30)."""
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_perturbation(pil_img: Image.Image, condition: str, cfg=None) -> Image.Image:
    """
    Apply any of the 9 simulated adverse weather / imaging conditions.

    Args:
        pil_img: Input RGB PIL Image.
        condition: One of 'clean', 'gaussian_noise', 'salt_pepper', 'blur',
                   'brightness_down', 'brightness_up', 'contrast_low',
                   'contrast_high', 'fog', 'rain', 'jpeg_compression'.
        cfg: Optional configuration with specific parameter values.

    Returns:
        Perturbed PIL Image.
    """
    if condition in ("clean", "none", None):
        return pil_img

    c = condition.lower().strip()
    if c == "gaussian_noise":
        std = cfg.robustness.gaussian_noise_std if cfg else 0.05
        return apply_gaussian_noise(pil_img, std=std)
    elif c in ("salt_pepper", "salt_and_pepper"):
        amt = cfg.robustness.salt_pepper_amount if cfg else 0.02
        return apply_salt_pepper_noise(pil_img, amount=amt)
    elif c == "blur":
        rad = cfg.robustness.blur_radius if cfg else 3.0
        return apply_blur(pil_img, radius=rad)
    elif c == "brightness_down":
        factor = cfg.robustness.brightness_down_factor if cfg else 0.70
        return apply_brightness(pil_img, factor=factor)
    elif c == "brightness_up":
        factor = cfg.robustness.brightness_up_factor if cfg else 1.30
        return apply_brightness(pil_img, factor=factor)
    elif c == "contrast_low":
        factor = cfg.robustness.contrast_low_factor if cfg else 0.70
        return apply_contrast(pil_img, factor=factor)
    elif c == "contrast_high":
        factor = cfg.robustness.contrast_high_factor if cfg else 1.30
        return apply_contrast(pil_img, factor=factor)
    elif c == "fog":
        op = cfg.robustness.fog_opacity if cfg else 0.40
        return apply_fog(pil_img, opacity=op)
    elif c == "rain":
        streaks = cfg.robustness.rain_num_streaks if cfg else 400
        return apply_rain(pil_img, num_streaks=streaks)
    elif c in ("jpeg_compression", "jpeg"):
        q = cfg.robustness.jpeg_quality if cfg else 30
        return apply_jpeg_compression(pil_img, quality=q)
    else:
        raise ValueError(f"Unknown perturbation condition: '{condition}'")
