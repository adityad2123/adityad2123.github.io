import numpy as np
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import cv2
from align_image_code import align_images
from skimage.draw import polygon, ellipse


def convolve4loops(img, kernel):
    # Convolution implementation with 4 for loops
    assert kernel.ndim == 2 and kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    height_padding = (kernel.shape[0] - 1) // 2
    width_padding = (kernel.shape[1] - 1) // 2
    height = img.shape[0]
    width = img.shape[1]
    padded_img = np.zeros((height + 2*height_padding, width + 2*width_padding))
    padded_img[height_padding:height_padding+height, width_padding:width_padding+width] = img
    flipped_kernel = kernel[::-1, ::-1]
    final_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            running_total = 0
            for k in range(0, kernel.shape[0]):
                for l in range(0, kernel.shape[1]):
                    running_total += padded_img[i+k, j+l] * flipped_kernel[k, l]
            final_img[i, j] = running_total
    return final_img

def convolve2loops(img, kernel):
    # Convolution implementation with 2 for loops
    assert kernel.ndim == 2 and kernel.shape[0] % 2 == 1 and kernel.shape[1] % 2 == 1
    height_padding = (kernel.shape[0] - 1) // 2
    width_padding = (kernel.shape[1] - 1) // 2
    height = img.shape[0]
    width = img.shape[1]
    padded_img = np.zeros((height + 2*height_padding, width + 2*width_padding))
    padded_img[height_padding:height_padding+height, width_padding:width_padding+width] = img
    flipped_kernel = kernel[::-1, ::-1]
    final_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            final_img[i, j] = (padded_img[i:i+kernel.shape[0], j:j+kernel.shape[1]] * flipped_kernel).sum()
    return final_img

# 1.1
me = img_as_float(io.imread("inputs/headshotsquare.jpeg", as_gray=True))

plt.imsave("out_p1/original_gray.jpg", me, cmap="gray", vmin=0, vmax=1)
box_filter = np.ones((9, 9)) / 81

start = time.perf_counter()
box4 = convolve4loops(me, box_filter)
print(f"4 loops: {time.perf_counter() - start:.4f}s")
plt.imsave("out_p1/box_4loops.jpg", box4, cmap="gray", vmin=0, vmax=1)

start = time.perf_counter()
box2 = convolve2loops(me, box_filter)
print(f"2 loops: {time.perf_counter() - start:.4f}s")
plt.imsave("out_p1/box_2loops.jpg", box2, cmap="gray", vmin=0, vmax=1)

start = time.perf_counter()
boxS = convolve2d(me, box_filter, mode="same", boundary="fill", fillvalue=0)
print(f"scipy: {time.perf_counter() - start:.4f}s")
plt.imsave("out_p1/box_scipy.jpg", boxS, cmap="gray", vmin=0, vmax=1)

print("The 4 for-loop implementation is sound.", np.allclose(box4, boxS))
print("The 2 for-loop implementation is sound.", np.allclose(box2, boxS))


def save_derivative(img, kernel, path, bound=None):
    # Convolve with a derivative filter and change the bounds so that 0 is gray
    out_p1 = convolve2d(img, kernel, mode="same", boundary="fill", fillvalue=0)
    bound = np.abs(out_p1).max() if bound is None else bound
    plt.imsave(path, out_p1, cmap="gray", vmin=-bound, vmax=bound)
    return out_p1

D_x = np.array([[1, 0, -1]])
D_y = np.array([[1], [0], [-1]])
me_D_x = save_derivative(me, D_x, "out_p1/me_D_x.jpg")
me_D_y = save_derivative(me, D_y, "out_p1/me_D_y.jpg")

# 1.2 
cameraman = img_as_float(io.imread("inputs/cameraman.png", as_gray=True))
cameraman_D_x = save_derivative(cameraman, D_x, "out_p1/cameraman_D_x.jpg")
cameraman_D_y = save_derivative(cameraman, D_y, "out_p1/cameraman_D_y.jpg")

grad_mag_cameraman = np.sqrt(cameraman_D_x**2 + cameraman_D_y**2)
plt.imsave("out_p1/cameraman_gradient.jpg", grad_mag_cameraman, cmap="gray", vmin=0, vmax=grad_mag_cameraman.max())

thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
fig, axes = plt.subplots(1, len(thresholds), figsize=(4 * len(thresholds), 4))
for ax, t in zip(axes, thresholds):
    ax.imshow(grad_mag_cameraman >= t, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"t = {t}")
    ax.axis("off")
plt.savefig("out_p1/thresholds.png")

# We choose threshold 0.25
plt.imsave("out_p1/cameraman_edges.jpg", grad_mag_cameraman >= 0.25, cmap="gray", vmin=0, vmax=1)

# 1.3
gaussian_1d = cv2.getGaussianKernel(ksize=9, sigma=(9-1)/6)
gaussian_2d = np.outer(gaussian_1d, gaussian_1d)

DoG_x = convolve2d(gaussian_2d, D_x, mode="same")
DoG_y = convolve2d(gaussian_2d, D_y, mode="same")

def show_filter(f, path, title=""):
    # Show the filter
    bound = np.abs(f).max()
    plt.figure(figsize=(4, 4))
    plt.imshow(f, cmap="gray", vmin=-bound, vmax=bound, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

show_filter(gaussian_2d, "out_p1/filter_G.jpg", "Gaussian")
show_filter(DoG_x, "out_p1/filter_DoG_x.jpg", "DoG_x")
show_filter(DoG_y, "out_p1/filter_DoG_y.jpg", "DoG_y")

# Method 1: Blur, then derivative
cameraman_blurred = convolve2d(cameraman, gaussian_2d, mode="same", boundary="fill", fillvalue=0)
plt.imsave("out_p1/cameraman_blurred.jpg", cameraman_blurred, cmap="gray", vmin=0, vmax=1)
cam_blur_D_x = save_derivative(cameraman_blurred, D_x, "out_p1/cam_blur_D_x.jpg", bound=0.53)
cam_blur_D_y = save_derivative(cameraman_blurred, D_y, "out_p1/cam_blur_D_y.jpg", bound=0.53)

# Method 2: #Doboth
cam_blur_D_x_g = save_derivative(cameraman, DoG_x, "out_p1/cam_blur_D_x_g.jpg", bound=0.53)
cam_blur_D_y_g = save_derivative(cameraman, DoG_y, "out_p1/cam_blur_D_y_g.jpg", bound=0.53)
p = gaussian_2d.shape[0]

grad_mag_cameraman_m1 = np.sqrt(cam_blur_D_x**2 + cam_blur_D_y**2)
plt.imsave("out_p1/cameraman_edges_method1.jpg", grad_mag_cameraman_m1 >= 0.125, cmap="gray", vmin=0, vmax=grad_mag_cameraman_m1.max())
grad_mag_cameraman_m2 = np.sqrt(cam_blur_D_x_g**2 + cam_blur_D_y_g**2)
plt.imsave("out_p1/cameraman_edges_method2.jpg", grad_mag_cameraman_m1 >= 0.125, cmap="gray", vmin=0, vmax=grad_mag_cameraman_m1.max())

# 2.1

# per lecture, we can think of it as f * ((1+alpha)e - alpha*g)
def unsharp_then_mask_kernel(alpha, size):
    gaussian_1d = cv2.getGaussianKernel(ksize=size, sigma=(size - 1) / 6)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    identity = np.zeros((size, size))
    identity[size // 2, size // 2] = 1
    return (1+alpha)*identity - (alpha)*gaussian_2d

def convolve_channels(img, kernel):
    # This is a 3D convolution, we convolve for each color channel separately
    conv = lambda ch: convolve2d(ch, kernel, mode="same", boundary="symm")
    return np.stack([conv(img[:, :, c]) for c in range(img.shape[2])], axis=-1)

images = {"taj": "inputs/taj.jpg", "pumpkin": "inputs/pumpkin.jpg"}
alphas = [0, 0.5, 1, 2, 5, 10]

for name, path in images.items():
    img = img_as_float(io.imread(path))
    for alpha in alphas:
        k = unsharp_then_mask_kernel(alpha, 9)
        out = np.clip(convolve_channels(img, k), 0, 1)
        plt.imsave(f"out_p2/{name}_alpha_{alpha}.jpg", out)

# we also need to show the blurred and high freq versions for an alpha that I like
for name, path in images.items():
    img = img_as_float(io.imread(path))
    size = 9
    gaussian_1d = cv2.getGaussianKernel(ksize=size, sigma=(size - 1) / 6)
    gaussian_2d = np.outer(gaussian_1d, gaussian_1d)
    blurred = convolve_channels(img, gaussian_2d)
    high = img - blurred
    plt.imsave(f"out_p2/{name}_blurred.jpg", np.clip(blurred, 0, 1))
    high_grayscale = high.mean(axis=-1)
    plt.imsave(f"out_p2/{name}_high.jpg", high_grayscale, cmap="gray", vmin=np.abs(high_grayscale).min(), vmax=np.abs(high_grayscale).max())


# 2.2

# First load images
telly = img_as_float(io.imread('inputs/telephone.jpg', as_gray=True))
banana = img_as_float(io.imread('inputs/banana.jpg', as_gray=True))
# Next align images (this code is provided, but may be improved)

puffin = img_as_float(io.imread('inputs/puffin.jpg', as_gray=True))
baseball = img_as_float(io.imread('inputs/baseball.jpg', as_gray=True))

derek = img_as_float(io.imread('inputs/derek.jpg', as_gray=True))
nutmeg = img_as_float(io.imread('inputs/nutmeg.jpg', as_gray=True))

# Next align images (this code is provided, but may be improved)
telly_aligned, banana_aligned = align_images(telly, banana)
plt.imsave(f"out_hybrid/telephone_aligned.jpg", telly_aligned, cmap='gray', vmin=0, vmax=1)
plt.imsave(f"out_hybrid/banana_aligned.jpg", banana_aligned, cmap='gray', vmin=0, vmax=1)
puffin_aligned, baseball_aligned = align_images(puffin, baseball)
derek_aligned, nutmeg_aligned = align_images(derek, nutmeg)


def hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2, fourier=False, name=""):
    # Produces hybrid images
    ksize1 = 2 * int(3 * sigma1) + 1
    g1 = cv2.getGaussianKernel(ksize=ksize1, sigma=sigma1)
    low_pass_gauss = np.outer(g1, g1)

    ksize2 = 2 * int(3 * sigma2) + 1
    g2 = cv2.getGaussianKernel(ksize=ksize2, sigma=sigma2)
    high_pass_gauss = np.outer(g2, g2)

    identity = np.zeros((ksize2, ksize2))
    identity[ksize2 // 2, ksize2 // 2] = 1
    low = convolve2d(im1_aligned, low_pass_gauss, mode="same", boundary="fill", fillvalue=0)
    high = convolve2d(im2_aligned, identity - high_pass_gauss, mode="same", boundary="fill", fillvalue=0)

    hybrid = low + high

    if fourier:
        images = {"im1": im1_aligned, "im2": im2_aligned, "im1_low": low, "im2_high": high, "hybrid": hybrid}
        for label, im in images.items():
            spectrum = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))))
            plt.imsave(f"out_hybrid/{name}_fourier_{label}.png", spectrum, cmap="gray")

    plt.imsave(f"out_hybrid/{name}_hybrid.jpg", hybrid, cmap="gray", vmin=0, vmax=1)
    return hybrid

bt_sigma1 = 7
bt_sigma2 = 4
bt_hybrid = hybrid_image(telly_aligned, banana_aligned, bt_sigma1, bt_sigma2, fourier=True, name='banana_telephone2')
pb_sigma1 = 15
pb_sigma2 = 7
pb_hybrid = hybrid_image(puffin_aligned, baseball_aligned, pb_sigma1, pb_sigma2, fourier=False, name='puffin_baseball')
dn_sigma1 = 15
dn_sigma2 = 7
dn_hybrid = hybrid_image(derek_aligned, nutmeg_aligned, dn_sigma1, dn_sigma2, fourier=False, name='derek_nutmeg')

# 2.3, 2.4

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

blend_images("oraple", apple, orange)
blend_images("bridges", golden, bay)
blend_images("cones", cream, traffic)
blend_images("eepy", dazed, sun, elliptical=True)