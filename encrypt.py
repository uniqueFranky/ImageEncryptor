import numpy as np
import random
import sequence
import utils
from registry import encryptor_registry, operation_registry, chaos_mapping_registry, transform_registry, sequence_registry
import trans
import operation

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


@encryptor_registry.register('BaseSequence')
class BaseSequenceEncryptor(BaseEncryptor):
    def __init__(self):
        super().__init__()
        self.sys = None
        self.ops = []
        self.total_steps = 0
    
    def add_chaos_map(self, map, initial):
        self.sys.add_mapping(map, inital=initial)
    
    def add_operation(self, op):
        if not isinstance(op, operation.BaseOperation):
            print(f'{op} not supported')
            return
        self.ops.append(op)
        
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


@encryptor_registry.register('BaseChaos')
class BaseChaosEncryptor(BaseSequenceEncryptor):
    def __init__(self):
        super().__init__()
        self.sys = sequence_registry.build('Chaos')


@encryptor_registry.register('ClassicChaos')
class ClassicChaosEncryptor(BaseSequenceEncryptor):
    def __init__(self, column_shuffle_times=3, row_shuffle_times=3, diffusion_times=3, compositional_times=3, 
                 arnold_a=1, arnold_b=1, arnold_initial=[1.2, 2.5],
                 tent_p=0.5, tent_initial=0.5):
        super().__init__()
        self.sys = sequence_registry.build('Chaos')
        # init chaos operations
        column_shuffle = operation_registry.build('ColumnShuffle', times=column_shuffle_times)
        row_shuffle = operation_registry.build('RowShuffle', times=row_shuffle_times)
        diffusion = operation_registry.build('Diffusion', times=diffusion_times)
        compositional = operation_registry.build('Compositional', [column_shuffle, row_shuffle, diffusion], times=compositional_times)
        self.add_operation(compositional)

        # init chaos mappings
        arnold = chaos_mapping_registry.build('Arnold', a=arnold_a, b=arnold_b)
        self.add_chaos_map(arnold, initial=arnold_initial)

        tent = chaos_mapping_registry.build('Tent', p=tent_p)
        self.add_chaos_map(tent, initial=tent_initial)

@encryptor_registry.register('DiscreteCosineChaos')
class DiscreteCosineChaos(BaseSequenceEncryptor):
    def __init__(self, column_shuffle_times=3, row_shuffle_times=3, compositional_times=3, 
                 arnold_a=1, arnold_b=1, arnold_initial=[1.2, 2.5],
                 tent_p=0.5, tent_initial=0.5):
        super().__init__()
        self.sys = sequence_registry.build('Chaos')
        # init DCT operation
        dct = operation_registry.build('DiscreteCosineTransform', times=1)
        self.add_operation(dct)

        # init chaos operations
        column_shuffle = operation_registry.build('ColumnShuffle', times=column_shuffle_times)
        row_shuffle = operation_registry.build('RowShuffle', times=row_shuffle_times)
        compositional = operation_registry.build('Compositional', [column_shuffle, row_shuffle], times=compositional_times)
        self.add_operation(compositional)

        # init chaos mappings
        arnold = chaos_mapping_registry.build('Arnold', a=arnold_a, b=arnold_b)
        self.add_chaos_map(arnold, initial=arnold_initial)

        tent = chaos_mapping_registry.build('Tent', p=tent_p)
        self.add_chaos_map(tent, initial=tent_initial)


@encryptor_registry.register('BaseRandom')
class BaseRandom(BaseSequenceEncryptor):
    def __init__(self, seed):
        super().__init__()
        self.sys = sequence_registry.build('Random', seed)
    

@encryptor_registry.register('ClassicRandom')
class ClassicRandom(BaseRandom):
    def __init__(self, seed, column_shuffle_times=3, row_shuffle_times=3, diffusion_times=3, compositional_times=3,):
        super().__init__(seed)
                # init chaos operations
        column_shuffle = operation_registry.build('ColumnShuffle', times=column_shuffle_times)
        row_shuffle = operation_registry.build('RowShuffle', times=row_shuffle_times)
        diffusion = operation_registry.build('Diffusion', times=diffusion_times)
        compositional = operation_registry.build('Compositional', [column_shuffle, row_shuffle, diffusion], times=compositional_times)
        self.add_operation(compositional)

        