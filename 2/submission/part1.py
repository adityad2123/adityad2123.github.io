import numpy as np
from scipy.signal import convolve2d
import time
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.transform import rescale
import cv2


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