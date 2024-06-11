import math
import numpy as np
import utils
import random
from registry import chaos_mapping_registry, sequence_registry


# 混沌映射基类
class BaseChaosMapping:
    def __call__(self, x):
        pass
    
    def __len__(self):  # 该映射所需要的初值数量
        pass


# Logistic映射
@chaos_mapping_registry.register('Logistic')
class LogisticMapping(BaseChaosMapping):
    def __init__(self, mu=0.5):
        self.mu = mu

    def __call__(self, x):
        return self.mu * x * (1 - x)
    
    def __len__(self):
        return 1


# Tent映射
@chaos_mapping_registry.register('Tent')
class TentMapping(BaseChaosMapping):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x / self.p if 0 <= x < self.p else (1 - x) / (1 - self.p)
    
    def __len__(self):
        return 1
    

# Arnold映射
@chaos_mapping_registry.register('Arnold')
class ArnoldMapping(BaseChaosMapping):
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b
    
    def __call__(self, v):
        x = v[0]
        y = v[1]
        newx = x + self.a * y
        newy = self.b * x + (self.a * self.b + 1) * y
        newx, _ = math.modf(newx)
        newy, _ = math.modf(newy)
        return [newx, newy]
    
    def __len__(self):  # Arnold映射需要两个初值
        return 2


# 序列发生器基类
class BaseSequenceSystem:
    # 获取长度为length的序列
    def get_sequence(self, length=100):
        pass

    # 重制序列发生器的状态
    def reset(self):
        pass

    # 获取下一个数值
    def __next__(self):
        pass

    # 获取长度为length的反向的迭代器
    def get_reverse_iterator(self, length):
        pass


# 混沌系统，继承自序列发生器基类
@sequence_registry.register('Chaos')
class ChaosSystem(BaseSequenceSystem):
    def __init__(self):
        self.map_list = []  # 混沌映射
        self.inital_value = []  # 混沌初值
        self.current_status = []  # 混沌状态

    def add_mapping(self, map: BaseChaosMapping, inital):  # 添加混沌映射
        if inital is list and len(inital) != len(map):
            print(f'mapping needs {len(map)} initial values, but {len(inital)} provided')
        self.map_list.append(map)

        # 每个映射都对应有其初值和状态
        self.inital_value.append(inital)
        self.current_status.append(inital)
    

    def get_sequence(self, length=100):
        '''
        从initial_value开始，计算混沌序列，共生成length个
        不影响current_status
        返回的数组的shape为[length, num_maps]
        '''
        result = []
        chaos = self.inital_value[:]
        for _ in range(length):
            for (i, map) in enumerate(self.map_list):
                chaos[i] = map(chaos[i])
            
            # extract_element将从生成多个值的混沌映射中取出一个值
            # 比如Arnold映射将会产生两个值，这里取第一个值作为混沌序列中的元素
            result.append(utils.extract_element(chaos))
        return result


    def __next__(self):
        '''
        以current_status为当前状态，生成下一个混沌值，并更新状态
        '''
        for (i, map) in enumerate(self.map_list):
            self.current_status[i] = map(self.current_status[i])
        return utils.extract_element(self.current_status)
    
    def get_next(self):
        return next(self)
    
    def reset(self):  # 重置混沌状态为初值状态
        self.current_status = self.inital_value[:]

    def get_reverse_iterator(self, length):  # 获取反向序列发生器
        self.reset()
        seq = reversed(self.get_sequence(length))
        return iter(seq)
        

# 随机系统，继承自序列发生器基类
@sequence_registry.register('Random')
class RandomSystem(BaseSequenceSystem):
    def __init__(self, seed):
        self.seed = seed

    def get_sequence(self, length=100):
        result = []
        for _ in range(length):
            result.append(random.randint(0, 2**15))
        return result
    
    def reset(self):
        random.seed(self.seed)

    def get_reverse_iterator(self, length):
        self.reset()
        seq = reversed(self.get_sequence(length))
        return iter(seq)
    
    def __next__(self):
        return random.randint(0, 2**15)
        
