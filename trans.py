import numpy as np
import pywt
import scipy
from registry import transform_registry

class BaseTransform:

    @staticmethod
    def forward(rgb):
        pass

    @staticmethod
    def backward(rgb):
        pass


@transform_registry.register('RawDiscreteCosineTransform')
class DiscreteCosineTransform(BaseTransform):
    def dct_2d(self, block):
        N = block.shape[0]
        dct_matrix = np.zeros((N, N), dtype=float)
        for u in range(N):
            for v in range(N):
                sum = 0.0
                for x in range(N):
                    for y in range(N):
                        sum += block[x, y] * np.cos((2*x + 1) * u * np.pi / (2 * N)) * np.cos((2*y + 1) * v * np.pi / (2 * N))
                if u == 0:
                    cu = np.sqrt(1 / N)
                else:
                    cu = np.sqrt(2 / N)
                if v == 0:
                    cv = np.sqrt(1 / N)
                else:
                    cv = np.sqrt(2 / N)
                dct_matrix[u, v] = cu * cv * sum
        return dct_matrix

    def idct_2d(self, block):
        N = block.shape[0]
        idct_matrix = np.zeros((N, N), dtype=float)
        for x in range(N):
            for y in range(N):
                sum = 0.0
                for u in range(N):
                    for v in range(N):
                        if u == 0:
                            cu = np.sqrt(1 / N)
                        else:
                            cu = np.sqrt(2 / N)
                        if v == 0:
                            cv = np.sqrt(1 / N)
                        else:
                            cv = np.sqrt(2 / N)
                        sum += cu * cv * block[u, v] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                idct_matrix[x, y] = sum
        return idct_matrix

    def block_transform(self, image, block_size, transform_func):
        h, w = image.shape

        # 计算需要的填充量
        h_pad = (block_size - (h % block_size)) % block_size
        w_pad = (block_size - (w % block_size)) % block_size

        # 对图片进行填充
        padded_image = np.pad(image, ((0, h_pad), (0, w_pad)), mode='constant', constant_values=0)

        # 进行变换
        padded_h, padded_w = padded_image.shape
        transformed_image = np.zeros((padded_h, padded_w), dtype=float)
        for i in range(0, padded_h, block_size):
            for j in range(0, padded_w, block_size):
                block = padded_image[i:i+block_size, j:j+block_size]
                transformed_image[i:i+block_size, j:j+block_size] = transform_func(block)

        # 去掉填充部分
        transformed_image = transformed_image[:h, :w]
        return transformed_image

    def forward(self, rgb):
        transformed_rgb = np.copy(rgb).astype(np.float)
        for layer_id in range(rgb.shape[2]):
            layer = rgb[:, :, layer_id]
            transformed_rgb[:, :, layer_id] = self.block_transform(layer, 8, DiscreteCosineTransform.dct_2d)
        return transformed_rgb

    def backward(self, rgb):
        transformed_rgb = np.copy(rgb).astype(np.float)
        for layer_id in range(rgb.shape[2]):
            layer = rgb[:, :, layer_id]
            transformed_rgb[:, :, layer_id] = self.block_transform(layer, 8, DiscreteCosineTransform.idct_2d)
        return transformed_rgb
        

@transform_registry.register('DiscreteCosineTransform')
class ScipyDiscreteCosineTransform(BaseTransform):
    def dct_2d(self, image):
        return scipy.fftpack.dct(scipy.fftpack.dct(image.T, norm='ortho').T, norm='ortho')

    def idct_2d(self, dct_image):
        return scipy.fftpack.idct(scipy.fftpack.idct(dct_image.T, norm='ortho').T, norm='ortho')

    def forward(self, rgb):
        transformed_rgb = np.zeros_like(rgb, dtype=float)
        for layer_id in range(rgb.shape[2]):
            layer = rgb[:, :, layer_id]
            transformed_rgb[:, :, layer_id] = self.dct_2d(layer)
        return transformed_rgb

    def backward(self, transformed_rgb):
        reconstructed_rgb = np.zeros_like(transformed_rgb, dtype=float)
        for layer_id in range(transformed_rgb.shape[2]):
            layer = transformed_rgb[:, :, layer_id]
            reconstructed_rgb[:, :, layer_id] = self.idct_2d(layer)
        return reconstructed_rgb


@transform_registry.register('FourierTransform')
class FourierTransform(BaseTransform):
    def fft_2d(self, image):
        return np.fft.fft2(image)

    def ifft_2d(self, freq_domain_image):
        return np.fft.ifft2(freq_domain_image)

    def forward(self, rgb):
        transformed_rgb = np.zeros_like(rgb, dtype=complex)
        for layer_id in range(rgb.shape[2]):
            layer = rgb[:, :, layer_id]
            transformed_rgb[:, :, layer_id] = self.fft_2d(layer)
        return transformed_rgb

    def backward(self, transformed_rgb):
        reconstructed_rgb = np.zeros_like(transformed_rgb, dtype=float)
        for layer_id in range(transformed_rgb.shape[2]):
            layer = transformed_rgb[:, :, layer_id]
            reconstructed_rgb[:, :, layer_id] = np.abs(self.ifft_2d(layer))
        return reconstructed_rgb

