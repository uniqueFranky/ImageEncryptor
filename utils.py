from PIL import Image
import numpy as np
import encrypt
import math

def read_rgb(path):
    image = Image.open(path)
    out = image.convert("RGB")
    return np.array(out)

def show_rgb(rgb):
    arr = np.asarray(rgb)
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