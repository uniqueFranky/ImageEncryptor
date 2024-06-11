from PIL import Image
import numpy as np
import encrypt
import math

def read_rgb(path):
    image = Image.open(path)
    out = image.convert("RGB")
    return np.array(out)

def show_rgb(rgb):
    arr = np.asarray(np.clip(rgb, 0, 255).astype(np.uint8))
    img = Image.fromarray(arr)
    img.show()

def extract_element(chaos):
    result = []
    for i in range(len(chaos)):
        if isinstance(chaos[i], list):
            result.append(chaos[i][0])
        else:
            result.append(chaos[i])
    return result

def discrete(timestep):
    sum = 0
    for v in timestep:
        sum += 256 * v
    return math.floor(256 * sum) % 256

def rgb_to_yuv(image_rgb):
    image_rgb_copy = image_rgb.copy().astype(np.float)
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.14713, -0.28886, 0.436],
                  [0.615, -0.51499, -0.10001]])
    image_yuv = np.dot(image_rgb_copy, m.T)
    image_yuv[:, :, 1:] += 128.0
    image_yuv = np.round(image_yuv).astype(np.uint8)
    return image_yuv

def yuv_to_rgb(image_yuv):
    image_yuv_copy = image_yuv.copy().astype(np.float)
    m = np.array([[1.0, 0.0, 1.13983],
                  [1.0, -0.39465, -0.58060],
                  [1.0, 2.03211, 0.0]])
    image_yuv_copy[:, :, 1:] -= 128.0
    image_rgb = np.dot(image_yuv_copy, m.T)
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return image_rgb