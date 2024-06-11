from PIL import Image
import numpy as np
import encrypt
import math


# 从文件读取图像并转换为RGB三通道的numpy数组
def read_rgb(path):
    image = Image.open(path)
    out = image.convert("RGB")
    return np.array(out)


# 展示RGB图像
def show_rgb(rgb):
    arr = np.asarray(np.clip(rgb, 0, 255).astype(np.uint8))  # 基于变换域的加密会返回浮点值，要先离散化
    img = Image.fromarray(arr)
    img.show()


# 若一个混沌映射产生多个状态值，则取其中的第一个状态值，产生混沌序列
def extract_element(chaos):
    result = []
    for i in range(len(chaos)):
        if isinstance(chaos[i], list):
            result.append(chaos[i][0])
        else:
            result.append(chaos[i])
    return result


# 对混沌序列进行离散化
def discrete(timestep):
    sum = 0
    if isinstance(timestep, list) or isinstance(timestep, np.ndarray):  # 传入的是混沌序列
        for v in timestep:  # 综合所有的混沌值，计算得到一个离散值
            sum += 256 * v
    else:  # 传入的是随机数
        sum = timestep
    return math.floor(256 * sum) % 256


# 把RGB表示的图像转换为YUV表示
def rgb_to_yuv(image_rgb):
    image_rgb_copy = image_rgb.copy().astype(np.float)
    m = np.array([[0.299, 0.587, 0.114],
                  [-0.14713, -0.28886, 0.436],
                  [0.615, -0.51499, -0.10001]])
    image_yuv = np.dot(image_rgb_copy, m.T)
    image_yuv[:, :, 1:] += 128.0
    image_yuv = np.round(image_yuv).astype(np.uint8)
    return image_yuv


# 把YUV表示的图像转换为RGB表示
def yuv_to_rgb(image_yuv):
    image_yuv_copy = image_yuv.copy().astype(np.float)
    m = np.array([[1.0, 0.0, 1.13983],
                  [1.0, -0.39465, -0.58060],
                  [1.0, 2.03211, 0.0]])
    image_yuv_copy[:, :, 1:] -= 128.0
    image_rgb = np.dot(image_yuv_copy, m.T)
    image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return image_rgb