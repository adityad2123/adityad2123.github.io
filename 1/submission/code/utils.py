import numpy as np
import skimage as sk
import warnings
warnings.filterwarnings('error')

def l2(img1, img2):
    # Calculates the Euclidean Distance between two images;
    # we want to minimize this, but our code is built for maximums, so we flip the sign
    return -1 * np.sqrt(np.sum((img1 - img2) ** 2))

def ncc(img1, img2):
    # Calculates the Normalized Cross Correlation between two images
    img1_centered = img1 - np.mean(img1)
    img2_centered = img2 - np.mean(img2)
    img1_mag = np.linalg.norm(img1_centered)
    img2_mag = np.linalg.norm(img2_centered)
    return np.sum(img1_centered * img2_centered) / (img1_mag * img2_mag)

def check_image(img1, img2, disp_x, disp_y, metric, crop=0.1):
    # Returns the metric we want to maximize calculated over the middle (1-2*crop) proportion of the image
    displaced_img2 = np.roll(img2, shift=(disp_x, disp_y), axis=(1, 0))
    height, width = img1.shape
    cropped_height, cropped_width = int(height * crop), int(width * crop)
    cropped_img1 = img1[cropped_height:height - cropped_height, cropped_width:width - cropped_width]
    cropped_img2 = displaced_img2[cropped_height:height - cropped_height, cropped_width:width - cropped_width]

    result = metric(cropped_img1, cropped_img2)
    return result

def align(img1, img2, disp_x, disp_y, start_x, start_y, metric):
    # Aligns the image by searching over a window of start_x +/- disp_x, start_y +/- disp_y and choosing (x, y) to maximize metric
    best_disp = (0, 0)
    best_val = float('-inf')
    for i in range(disp_x * -1 + start_x, disp_x + start_x + 1):
        for j in range(disp_y * -1 + start_y, disp_y + start_y + 1):
            val = check_image(img1, img2, i, j, metric)
            if val > best_val:
                best_val = val
                best_disp = (i, j)
    return best_disp

def align_recursive(img1, img2, max_disp_x, max_disp_y, metric):
    # Recursively aligns the image with max_disp_x and max_disp_y as the maximum displacements to search over for the corresponding images
    if img1.shape[0] < 500:
        return align(img1, img2, max_disp_x, max_disp_y, 0, 0, metric)
    height = img1.shape[0] 
    width = img1.shape[1]
    x, y = align_recursive(sk.transform.resize(img1, (height // 2, width // 2), anti_aliasing=True),
                    sk.transform.resize(img2, (height // 2, width // 2), anti_aliasing=True),
                    max_disp_x * 2, max_disp_y * 2, metric)
    return align(img1, img2, max_disp_x, max_disp_y, x * 2, y * 2, metric)