import numpy as np
import time
from skimage.metrics import structural_similarity as ssim
from registry import metric_registry


class Timer:
    def __init__(self):
        self.time_stamp = time.time()
    
    def stop(self):
        now = time.time()
        return now - self.time_stamp


class BaseMetric:
    def __call__(self, rgb1, rgb2):
        pass


@metric_registry.register('MSE')
class MeanSquaredErrorMetric(BaseMetric):
    def __call__(self, rgb1, rgb2):
        rgb1 = np.clip(rgb1, 0, 255).astype(np.uint8)
        rgb2 = np.clip(rgb2, 0, 255).astype(np.uint8)
        return np.mean((rgb1 - rgb2) ** 2)
    

@metric_registry.register('PSNR')
class PeakSignal2NoiseRatioMetric(BaseMetric):
    def __call__(self, rgb1, rgb2):
        rgb1 = np.clip(rgb1, 0, 255).astype(np.uint8)
        rgb2 = np.clip(rgb2, 0, 255).astype(np.uint8)
        mse = np.mean((rgb1 - rgb2) ** 2)
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr


@metric_registry.register('SSIM')
class StructuralSimilarityIndexMetric(BaseMetric):
    def __call__(self, rgb1, rgb2):
        if rgb1.shape != rgb2.shape:
            raise ValueError("Input images must have the same dimensions.")
        rgb1 = np.clip(rgb1, 0, 255).astype(np.uint8)
        rgb2 = np.clip(rgb2, 0, 255).astype(np.uint8)
        ssim_value = ssim(rgb1, rgb2, channel_axis=2)
        return ssim_value
