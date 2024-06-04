import numpy as np

class BaseEncryptor:
    def __init__(self):
        pass
    
    def encrypt(self, rgb):
        pass
    
    def decrypt(self, rgb):
        pass


class ArnoldEncryptor(BaseEncryptor):
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
    
    
