from PIL import Image
import numpy as np
import encrypt

def read_rgb(path):
    image = Image.open(path)
    out = image.convert("RGB")
    return np.array(out)

def show_rgb(rgb):
    arr = np.asarray(rgb)
    img = Image.fromarray(arr)
    img.show()