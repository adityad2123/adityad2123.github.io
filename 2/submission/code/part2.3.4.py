import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import cv2
from skimage.draw import polygon, ellipse

orange = img_as_float(io.imread("inputs/orange.jpeg"))
apple = img_as_float(io.imread("inputs/apple.jpeg"))

bay = img_as_float(io.imread("inputs/bay.png"))[:, :, :3]
golden = img_as_float(io.imread("inputs/golden.png"))[:, :, :3]
traffic = img_as_float(io.imread("inputs/traffic.png"))[:, :, :3]
cream = img_as_float(io.imread("inputs/icecream.png"))[:, :, :3]
sun = img_as_float(io.imread("inputs/sun.png"))[:, :, :3]
dazed = img_as_float(io.imread("inputs/dazed.png"))[:, :, :3]


def convolve_channels(img, kernel):
    conv = lambda ch: convolve2d(ch, kernel, mode="same", boundary="symm")
    return np.stack([conv(img[:, :, c]) for c in range(img.shape[2])], axis=-1)

def gaussian_stack(img, levels, size):
    stack = [img]
    gaussian_1d = cv2.getGaussianKernel(ksize=size, sigma=(size - 1) / 6)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    for i in range(1, levels):
        stack.append(convolve_channels(stack[-1], gaussian_2d))
    return stack

def laplacian_stack(g_stack):
    stack = []
    for i in range(0, len(g_stack) - 1):
        stack.append(g_stack[i] - g_stack[i+1])
    stack.append(g_stack[-1])
    return stack

def show_stacks(g_stack, l_stack, path, step=3):
    n = len(g_stack)
    shown = list(range(0, n, step))
    if shown[-1] != n - 1:
        shown.append(n - 1) 
    fig, axes = plt.subplots(2, len(shown), figsize=(3 * len(shown), 6))
    for col, i in enumerate(shown):
        axes[0, col].imshow(g_stack[i])
        lap = l_stack[i] if i == n - 1 else show_signed(l_stack[i])
        axes[1, col].imshow(lap)
        axes[0, col].set_title(f"G{i}")
        axes[1, col].set_title(f"L{i}")
    for ax in axes.ravel():
        ax.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

def show_signed(x):
    m = np.abs(x).max()
    return np.clip(x / (2 * m) + 0.5, 0, 1)

def make_mask(img, elliptical=False):
    mask = np.zeros(img.shape[:2])
    plt.imshow(img)
    if elliptical:
        (x1, y1), (x2, y2) = plt.ginput(2, timeout=0)
        plt.close()
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        radius_x, radius_y = abs(x2 - x1) / 2, abs(y2 - y1) / 2
        rows, cols = ellipse(center_y, center_x, radius_y, radius_x, shape=mask.shape)
    else:
        pts = np.array(plt.ginput(n=-1, timeout=0))
        plt.close()
        rows, cols = polygon(pts[:, 1], pts[:, 0], shape=mask.shape)
    mask[rows, cols] = 1
    return np.stack([mask] * 3, axis=-1)

def blend(img_a, img_b, mask, levels, size):
    a_g = gaussian_stack(img_a, levels, size)
    b_g = gaussian_stack(img_b, levels, size)
    a_l, b_l = laplacian_stack(a_g), laplacian_stack(b_g)
    m_g = gaussian_stack(mask, levels, size)
    blended_levels = [m_g[l] * a_l[l] + (1 - m_g[l]) * b_l[l] for l in range(levels)]
    blended = np.clip(sum(blended_levels), 0, 1)
    return blended, (a_g, a_l, b_g, b_l, m_g)

def show_blend_figure(a_l, b_l, m_g, path):
    rows = [0, 5, 10]
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    levels = len(a_l)
    for r, l in enumerate(rows):
        a = m_g[l] * a_l[l]
        b = (1 - m_g[l]) * b_l[l]
        for c, im in enumerate([a, b, a + b]):
            axes[r, c].imshow(show_signed(im))
    a_half = sum(m_g[l] * a_l[l] for l in range(levels))
    b_half = sum((1 - m_g[l]) * b_l[l] for l in range(levels))
    for c, im in enumerate([a_half, b_half, a_half + b_half]):
        axes[3, c].imshow(np.clip(im, 0, 1))
    for ax in axes.ravel():
        ax.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)

def blend_images(name, img_a, img_b, levels=15, size=23, elliptical=False):

    mask = make_mask(img_a, elliptical)
    plt.imsave(f"out_multi/{name}_mask.jpg", mask)

    blended, (a_g, a_l, b_g, b_l, m_g) = blend(img_a, img_b, mask, levels, size)
    plt.imsave(f"out_multi/{name}_blend.jpg", blended)
    show_stacks(a_g, a_l, f"out_multi/{name}_stack_a.jpg")
    show_stacks(b_g, b_l, f"out_multi/{name}_stack_b.jpg")
    show_blend_figure(a_l, b_l, m_g, f"out_multi/{name}_figure.jpg")
    return blended

# blend_images("oraple", apple, orange)
# blend_images("bridges", golden, bay)
# blend_images("cones", cream, traffic)
blend_images("eepy", dazed, sun, elliptical=True)