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
