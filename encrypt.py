import numpy as np
import random
import chaos
import utils
from registry import encryptor_registry, chaos_operation_registry

class BaseEncryptor:
    def __init__(self):
        pass
    
    def encrypt(self, rgb):
        pass
    
    def decrypt(self, rgb):
        pass


@encryptor_registry.register('Arnold')
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

@encryptor_registry.register('BaseChaos')
class BaseChaosTransform(BaseEncryptor):
    def __init__(self):
        super().__init__()
        self.sys = chaos.ChaosSystem()
        self.ops = []
        self.total_steps = 0
    
    def add_chaos_map(self, map, initial):
        self.sys.add_mapping(map, inital=initial)
    
    def add_operation(self, operation):
        if not isinstance(operation, BaseChaosOperation):
            print(f'{operation} not supported')
            return
        self.ops.append(operation)
        
    def encrypt(self, rgb):
        result = rgb.copy()
        for op in self.ops:
            result = op(result, self.sys, reverse=False)
        return result
    
    def decrypt(self, rgb):
        result = rgb.copy()
        self.total_steps = 0
        for op in self.ops:
            self.total_steps += op.get_cost(rgb)
        it = self.sys.get_reverse_iterator(self.total_steps)
        for op in reversed(self.ops):
            result = op(result, it, reverse=True)
        return result


class BaseChaosOperation:
    def __init__(self, times=1):
        self.times = times

    def __call__(cls, rgb, it: iter, reverse=False):
        pass

    def get_cost(cls, rgb):
        pass


@chaos_operation_registry.register('RowShuffle')
class RowShuffleOperation(BaseChaosOperation):
    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):
            for dim in (range(rgb.shape[2]) if not reverse else reversed(range(rgb.shape[2]))):
                x1 = utils.discrete(next(it)) % rgb.shape[0]
                x2 = utils.discrete(next(it)) % rgb.shape[0]
                rgb[[x1, x2], :, dim] = rgb[[x2, x1], :, dim]
        return rgb

    def get_cost(self, rgb):
        return 2 * rgb.shape[2] * self.times


@chaos_operation_registry.register('ColumnShuffle')
class ColumnShuffleOperation(BaseChaosOperation):
    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):
            for dim in (range(rgb.shape[2]) if not reverse else reversed(range(rgb.shape[2]))):
                y1 = utils.discrete(next(it)) % rgb.shape[1]
                y2 = utils.discrete(next(it)) % rgb.shape[1]
                rgb[:, [y1, y2], dim] =rgb[:, [y2, y1], dim]
        return rgb

    def get_cost(self, rgb):
        return 2 * rgb.shape[2] * self.times


@chaos_operation_registry.register('Diffusion')
class DiffusionOperation(BaseChaosOperation):
    def __call__(self, rgb, it: iter, reverse=False):
        shape = rgb.shape
        flt = rgb.flatten()
        for _ in range(self.times):
            if not reverse:
                for i in range(len(flt)):
                    if i == 0:
                        flt[i] = (flt[i] + utils.discrete(next(it))) % 256
                    else:
                        flt[i] = (flt[i - 1] + flt[i] + utils.discrete(next(it))) % 256
            else:
                for i in reversed(range(len(flt))):
                    if i == 0:
                        flt[i] = (flt[i] - utils.discrete(next(it))) % 256
                    else:
                        flt[i] = (flt[i] - flt[i - 1] - utils.discrete(next(it))) % 256
        return flt.reshape(shape)
    

    def get_cost(self, rgb):
        return rgb.shape[0] * rgb.shape[1] * rgb.shape[2] * self.times


@chaos_operation_registry.register('Compositional')
class CompositionalChaosOperation(BaseChaosOperation):
    def __init__(self, op_list, times=1):
        self.op_list = op_list
        self.times = times
    
    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):
            if not reverse:
                for op in self.op_list:
                    rgb = op(rgb, it, reverse)
            else:
                for op in reversed(self.op_list):
                    rgb = op(rgb, it, reverse)
        return rgb

    def get_cost(self, rgb):
        cnt = 0
        for op in self.op_list:
            cnt += op.get_cost(rgb)
        return cnt * self.times

