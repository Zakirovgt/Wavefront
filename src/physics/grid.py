import numpy as np


def block_average_downsample(img: np.ndarray, factor: int):
    H, W = img.shape
    if H % factor != 0 or W % factor != 0:
        raise ValueError("Размер изображения должен делиться на factor")
    return img.reshape(H // factor, factor, W // factor, factor).mean(axis=(1, 3))


def build_grid(size: int):
    xs = np.linspace(-1, 1, size, dtype=np.float32)
    ys = np.linspace(-1, 1, size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)
    inside = (R <= 1.0)
    step = 2.0 / (size - 1)
    return xs, ys, X, Y, R, Theta, inside, step
