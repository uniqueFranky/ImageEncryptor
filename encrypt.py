import numpy as np
import random

class BaseEncryptor:
    def __init__(self):
        pass
    
    def encrypt(self, rgb):
        pass
    
    def decrypt(self, rgb):
        pass


class ArnoldTransform(BaseEncryptor):
    def __init__(self, a=1, b=1, shuffle_times=1):
        self.a = a
        self.b = b
        self.shuffle_times = shuffle_times
    
    def encrypt(self, rgb):
        N = rgb.shape[0]
        if rgb.shape[0] != rgb.shape[1]:
            print('Arnold only accepts images with same height and width')
            exit(-1)
        result = np.zeros(shape=rgb.shape, dtype=np.uint8)
        for _ in range(self.shuffle_times):
            for i in range(N):
                for j in range(N):
                    x = (i + self.b * j) % N
                    y = (self.a * i + (self.a * self.b + 1) * j) % N
                    result[x, y, :] = rgb[i, j, :]
        return result
    
    def decrypt(self, rgb):
        N = rgb.shape[0]
        if rgb.shape[0] != rgb.shape[1]:
            print('Arnold only accepts images with same height and width')
            exit(-1)
        result = np.zeros(shape=rgb.shape, dtype=np.uint8)
        for _ in range(self.shuffle_times):
            for i in range(N):
                for j in range(N):
                    x = ((self.a * self.b + 1) * i - self.b * j) % N
                    y = (-self.a * i + j) % N
                    result[x, y, :] = rgb[i, j, :]
        return result
    
    
class DiscreteCosineTransform(BaseEncryptor):
    def __init__(self):
        super().__init__()
        self.seed = random.random()

    def encrypt(self, rgb):
        N = rgb.shape[0]
        if rgb.shape[0] != rgb.shape[1]:
            print('DiscreteCosineTransform only accepts images with same height and width')
            exit(-1)
        result = np.zeros(shape=rgb.shape, dtype=np.uint8)
        random.seed(self.seed)
        for dim in range(rgb.shape[2]):
            old_layer = rgb[:, :, dim]
            new_layer = np.zeros(shape=rgb.shape[:-1])
            for i in range(N):
                for j in range(N):
                    if i == 0:
                        a = np.sqrt(1 / N)
                    else:
                        a = np.sqrt(2 / N)
                    new_layer[i, j] = a * np.cos(np.pi * (j + 0.5) * i / N)
            result[:, :, dim] = new_layer.dot(old_layer).dot(new_layer.T)
        return result
    
    def decrypt(self, rgb):
        N = rgb.shape[0]
        if rgb.shape[0] != rgb.shape[1]:
            print('DiscreteCosineTransform only accepts images with same height and width')
            exit(-1)
        result = np.zeros(shape=rgb.shape, dtype=np.uint8)
        random.seed(self.seed)
        for dim in range(rgb.shape[2]):
            old_layer = rgb[:, :, dim]
            new_layer = np.zeros(shape=rgb.shape[:-1])
            for i in range(N):
                for j in range(N):
                    if i == 0:
                        a = np.sqrt(1 / N)
                    else:
                        a = np.sqrt(2 / N)
                    new_layer[i, j] = a * np.cos(np.pi * (j + 0.5) * i / N)
            result[:, :, dim] = (new_layer.T).dot(old_layer).dot(new_layer)
        return result