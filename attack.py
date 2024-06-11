import numpy as np
import random
from registry import attacker_registry

class BaseAttacker:  # Attacker的基类
    def __init__(self, times=1):
        self.times = times

    def __call__(self, rgb):
        pass


@attacker_registry.register('PointReplace')
class PointReplaceAttacker(BaseAttacker):  # 随机替换一个像素点
    def __call__(self, rgb):
        attacked = rgb.copy()
        for _ in range(self.times):  # 执行self.times次替换
            x = random.randint(0, rgb.shape[0] - 1)
            y = random.randint(0, rgb.shape[1] - 1)
            z = random.randint(0, rgb.shape[2] - 1)
            attacked[x, y, z] = random.randint(0, 255)
        return attacked
    
@attacker_registry.register('RowErase')
class RowEraseAttacker(BaseAttacker):  # 随机擦除一行，即把一行置黑
    def __call__(self, rgb):
        attacked = rgb.copy()
        for _ in range(self.times):  # 执行self.times次替换
            x = random.randint(0, rgb.shape[0] - 1)
            attacked[x, :, :] = np.zeros((rgb.shape[1], rgb.shape[2]))
        return attacked


@attacker_registry.register('ColumnErase')
class RowEraseAttacker(BaseAttacker):  # 随机擦除一列，即把一列置黑
    def __call__(self, rgb):
        attacked = rgb.copy()
        for _ in range(self.times):  # 执行self.times次替换
            y = random.randint(0, rgb.shape[1] - 1)
            attacked[:, y, :] = np.zeros((rgb.shape[0], rgb.shape[2]))
        return attacked


@attacker_registry.register('BlockSwap')
class BlockSwapAttacker(BaseAttacker):  # 将图像的两个块进行交换
    def __init__(self, times=1, block_size=10):
        super().__init__(times)
        self.block_size = block_size

    def __call__(self, rgb):
        attacked = rgb.copy()
        for _ in range(self.times):  # 执行self.times次交换
            h, w, c = rgb.shape
            # 随机选择两个块的位置
            x1 = random.randint(0, h - self.block_size)
            y1 = random.randint(0, w - self.block_size)
            x2 = random.randint(0, h - self.block_size)
            y2 = random.randint(0, w - self.block_size)

            # 提取两个块
            block1 = attacked[x1:x1 + self.block_size, y1:y1 + self.block_size, :].copy()
            block2 = attacked[x2:x2 + self.block_size, y2:y2 + self.block_size, :].copy()

            # 交换两个块
            attacked[x1:x1 + self.block_size, y1:y1 + self.block_size, :] = block2
            attacked[x2:x2 + self.block_size, y2:y2 + self.block_size, :] = block1

        return attacked