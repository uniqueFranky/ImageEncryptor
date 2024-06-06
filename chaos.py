import math
import numpy as np
import utils
from registry import chaos_mapping_registry

class BaseChaosMapping:
    def __call__(self, x):
        pass
    
    def __len__(self):
        pass


@chaos_mapping_registry.register('Logistic')
class LogisticMapping(BaseChaosMapping):
    def __init__(self, mu=0.5):
        self.mu = mu

    def __call__(self, x):
        return self.mu * x * (1 - x)
    
    def __len__(self):
        return 1


@chaos_mapping_registry.register('Tent')
class TentMapping(BaseChaosMapping):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        return x / self.p if 0 <= x < self.p else (1 - x) / (1 - self.p)
    
    def __len__(self):
        return 1
    

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
    
    def __len__(self):
        return 2

class ChaosSystem:
    def __init__(self):
        self.map_list = []
        self.inital_value = []
        self.current_status = []

    def add_mapping(self, map: BaseChaosMapping, inital):
        if inital is list and len(inital) != len(map):
            print(f'mapping needs {len(map)} initial values, but {len(inital)} provided')
        self.map_list.append(map)
        self.inital_value.append(inital)
        self.current_status.append(inital)
    
    '''
    从initial_value开始，计算混沌序列，共生成length个
    返回的结果为[length, num_map]，每个map都会生成一个timestep中的一个元素
    不影响current_status
    '''
    def get_chaos_sequence(self, length=100):
        result = []
        chaos = self.inital_value[:]
        for _ in range(length):
            for (i, map) in enumerate(self.map_list):
                chaos[i] = map(chaos[i])
            result.append(utils.extract_element(chaos))
        return result

    '''
    以current_status为当前状态，生成下一个混沌值，并更新状态
    '''
    def __next__(self):
        for (i, map) in enumerate(self.map_list):
            self.current_status[i] = map(self.current_status[i])
        return utils.extract_element(self.current_status)
    
    def get_next_chaos(self):
        return next(self)
    
    def reset(self):
        self.current_status = self.inital_value[:]

    def get_reverse_iterator(self, length):
        seq = reversed(self.get_chaos_sequence(length))
        return iter(seq)
        

        
        
