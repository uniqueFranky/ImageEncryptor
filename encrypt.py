import numpy as np
import random
import sequence
import utils
from registry import encryptor_registry, operation_registry, chaos_mapping_registry, transform_registry, sequence_registry
import trans
import operation
import evaluate

# 用该装饰器来记录加密/解密时间
def before_encrypt(encrypt=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            t = evaluate.Timer()  # 创建计时器
            ret = func(*args, **kwargs)  # 执行加密/解密
            cost = t.stop()  # 停止计时器
            print(f'{args[0].__class__.__name__} took {cost} seconds to {"encrypt" if encrypt else "decrypt"}.')
            return ret
        return wrapper
    return decorator


# 加密器的基类
class BaseEncryptor:
    def __init__(self):
        pass
    
    def encrypt(self, rgb):  # 执行加密
        pass
    
    def decrypt(self, rgb):  # 执行解密
        pass


# 猫脸变换加密器
@encryptor_registry.register('Arnold')
class ArnoldTransform(BaseEncryptor):
    def __init__(self, a=1, b=1, shuffle_times=1):  # 传入猫脸变换的参数，以及执行次数
        self.a = a
        self.b = b
        self.shuffle_times = shuffle_times
    
    @before_encrypt(encrypt=True)
    def encrypt(self, rgb):
        N = rgb.shape[0]
        if rgb.shape[0] != rgb.shape[1]:
            print('Arnold only accepts images with same height and width')
            exit(-1)
        result = np.zeros(shape=rgb.shape, dtype=np.uint8)
        for _ in range(self.shuffle_times):
            for i in range(N):
                for j in range(N):  # 对图像的像素重新置位
                    x = (i + self.b * j) % N
                    y = (self.a * i + (self.a * self.b + 1) * j) % N
                    result[x, y, :] = rgb[i, j, :]
        return result
    
    @before_encrypt(encrypt=False)
    def decrypt(self, rgb):
        N = rgb.shape[0]
        if rgb.shape[0] != rgb.shape[1]:
            print('Arnold only accepts images with same height and width')
            exit(-1)
        result = np.zeros(shape=rgb.shape, dtype=np.uint8)
        for _ in range(self.shuffle_times):
            for i in range(N):
                for j in range(N):  # 逆向的置位
                    x = ((self.a * self.b + 1) * i - self.b * j) % N
                    y = (-self.a * i + j) % N
                    result[x, y, :] = rgb[i, j, :]
        return result


# 基于序列发生器的加密器
@encryptor_registry.register('BaseSequence')
class BaseSequenceEncryptor(BaseEncryptor):
    # 序列发生器是指：每次调用序列发生器时，其能够提供一个数值，用于下一步的加密/解密操作
    # 目前实现的序列发生器包括：混沌系统、随机系统
    def __init__(self):
        super().__init__()
        self.sys = None  # 序列发生器，可以是混沌系统/随机数发生器
        self.ops = []  # 要执行的加密操作
        self.total_steps = 0  # 执行所有加密操作需要序列发生器提供的数值数目
    
    def add_chaos_map(self, map, initial):  # 添加混沌系统所使用的映射函数
        self.sys.add_mapping(map, inital=initial)
    
    def add_operation(self, op):  # 添加要执行的加密操作，这些加密操作会根据序列发生器给出的数值来进行加密
        if not isinstance(op, operation.BaseOperation):
            print(f'{op} not supported')
            return
        self.ops.append(op)
        
    @before_encrypt(encrypt=True)
    def encrypt(self, rgb):  # 加密
        result = rgb.copy()
        for op in self.ops:  # 依次执行每个加密操作
            result = op(result, self.sys, reverse=False)
        return result
    
    @before_encrypt(encrypt=False)
    def decrypt(self, rgb):  # 解密
        result = rgb.copy()
        self.total_steps = 0
        for op in self.ops:  # 计算序列发生器需要产生多少个数值
            self.total_steps += op.get_cost(rgb)
        
        #  获取逆向的操作数序列，因为解密操作要按照加密操作的倒序来执行
        #  获取逆向操作数序列的方法是，先生成正向的序列，然后再反转
        it = self.sys.get_reverse_iterator(self.total_steps)
        for op in reversed(self.ops):  # 倒序执行每个加密操作的逆过程
            result = op(result, it, reverse=True)
        return result


# 基于混沌系统的加密器
@encryptor_registry.register('BaseChaos')
class BaseChaosEncryptor(BaseSequenceEncryptor):
    def __init__(self):
        super().__init__()
        self.sys = sequence_registry.build('Chaos')  # 设置序列发生器为混沌系统


# 预实现的基于混沌系统的加密器（即课件中所展示的）
@encryptor_registry.register('ClassicChaos')
class ClassicChaosEncryptor(BaseChaosEncryptor):
    # 将执行若干次[行置换, 列置换, 扩散]操作，然后再重复上述操作若干次
    # 使用的混沌映射为Arnold+Tent
    def __init__(self, column_shuffle_times=3, row_shuffle_times=3, diffusion_times=3, compositional_times=3, 
                 arnold_a=1, arnold_b=1, arnold_initial=[1.2, 2.5],
                 tent_p=0.5, tent_initial=0.5):
        super().__init__()

        # 添加加密操作
        column_shuffle = operation_registry.build('ColumnShuffle', times=column_shuffle_times)  # 列置换
        row_shuffle = operation_registry.build('RowShuffle', times=row_shuffle_times)  # 行置换
        diffusion = operation_registry.build('Diffusion', times=diffusion_times)  # 扩散
        compositional = operation_registry.build('Compositional', [column_shuffle, row_shuffle, diffusion], times=compositional_times)  # 三个操作的组合，重复若干次
        self.add_operation(compositional)

        # 添加混沌映射
        arnold = chaos_mapping_registry.build('Arnold', a=arnold_a, b=arnold_b)  # Arnold映射
        self.add_chaos_map(arnold, initial=arnold_initial)

        tent = chaos_mapping_registry.build('Tent', p=tent_p)  # Tent映射
        self.add_chaos_map(tent, initial=tent_initial)


# 预实现的、基于离散余弦变换的、基于混沌系统的加密器
# 即先对图像做离散余弦变换，然后再基于混沌系统在变换域上做加密操作
@encryptor_registry.register('DiscreteCosineChaos')
class DiscreteCosineChaos(BaseChaosEncryptor):
    def __init__(self, column_shuffle_times=3, row_shuffle_times=3, compositional_times=3, 
                 arnold_a=1, arnold_b=1, arnold_initial=[1.2, 2.5],
                 tent_p=0.5, tent_initial=0.5):
        super().__init__()

        # 添加离散余弦变换
        dct = operation_registry.build('DiscreteCosineTransform', times=1)
        self.add_operation(dct)

        # 添加在变换域上的加密操作
        column_shuffle = operation_registry.build('ColumnShuffle', times=column_shuffle_times)
        row_shuffle = operation_registry.build('RowShuffle', times=row_shuffle_times)
        compositional = operation_registry.build('Compositional', [column_shuffle, row_shuffle], times=compositional_times)
        self.add_operation(compositional)

        # 添加混沌系统的混沌映射
        arnold = chaos_mapping_registry.build('Arnold', a=arnold_a, b=arnold_b)
        self.add_chaos_map(arnold, initial=arnold_initial)

        tent = chaos_mapping_registry.build('Tent', p=tent_p)
        self.add_chaos_map(tent, initial=tent_initial)


# 基于随机序列发生器的加密器
@encryptor_registry.register('BaseRandom')
class BaseRandom(BaseSequenceEncryptor):
    def __init__(self, seed):  # 传入随机种子
        super().__init__()
        self.sys = sequence_registry.build('Random', seed)  # 设置序列发生器为随机系统
    

# 预实现的基于随机系统的加密器
@encryptor_registry.register('ClassicRandom')
class ClassicRandom(BaseRandom):
    def __init__(self, seed, column_shuffle_times=3, row_shuffle_times=3, diffusion_times=3, compositional_times=3,):
        super().__init__(seed)
        # 添加加密操作
        column_shuffle = operation_registry.build('ColumnShuffle', times=column_shuffle_times)  # 列置换
        row_shuffle = operation_registry.build('RowShuffle', times=row_shuffle_times)  # 行置换
        diffusion = operation_registry.build('Diffusion', times=diffusion_times)  # 扩散
        compositional = operation_registry.build('Compositional', [column_shuffle, row_shuffle, diffusion], times=compositional_times)
        self.add_operation(compositional)

