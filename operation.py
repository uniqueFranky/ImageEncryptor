from registry import operation_registry
import utils

class BaseOperation:  # 对图像（原始域或变换域）作加密操作的基类
    def __init__(self, times=1):
        self.times = times

    def __call__(cls, rgb, it: iter, reverse=False):
        '''
        it: 序列发生器
        reverse: 是否执行逆变换
        '''
        pass

    def get_cost(cls, rgb):  # 该操作需要从序列发生器获取多少个数值
        pass


@operation_registry.register('RowShuffle')
class RowShuffleOperation(BaseOperation):  # 随机交换两行，执行times次
    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):  # 执行times次
            for dim in (range(rgb.shape[2]) if not reverse else reversed(range(rgb.shape[2]))):  # 选择RGB三个通道之一
                # next(it)将从序列发生器获得一个值，再用utils.discrete把序列发生器得到的数值离散化
                x1 = utils.discrete(next(it)) % rgb.shape[0]  # 取模保证不越界
                x2 = utils.discrete(next(it)) % rgb.shape[0]
                rgb[[x1, x2], :, dim] = rgb[[x2, x1], :, dim]  # 交换两行
        return rgb

    def get_cost(self, rgb):
        return 2 * rgb.shape[2] * self.times


@operation_registry.register('ColumnShuffle')
class ColumnShuffleOperation(BaseOperation):  # 随机交换两列，执行times次。实现同RowShuffleOperation
    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):
            for dim in (range(rgb.shape[2]) if not reverse else reversed(range(rgb.shape[2]))):
                y1 = utils.discrete(next(it)) % rgb.shape[1]
                y2 = utils.discrete(next(it)) % rgb.shape[1]
                rgb[:, [y1, y2], dim] =rgb[:, [y2, y1], dim]
        return rgb

    def get_cost(self, rgb):
        return 2 * rgb.shape[2] * self.times


@operation_registry.register('Diffusion')
class DiffusionOperation(BaseOperation):  # 像素扩散操作，把一个像素的信息扩散到图像的其他部分
    def __call__(self, rgb, it: iter, reverse=False):
        shape = rgb.shape
        flt = rgb.flatten()  # 把二维图像展平为一维像素序列
        for _ in range(self.times):
            if not reverse:  # 执行正向扩散
                for i in range(len(flt)):  # 考虑原图像中的每个像素
                    if i == 0:  # 当前像素是图像中的第一个像素，该像素的信息将会被扩散到后面的所有像素
                        flt[i] = (flt[i] + utils.discrete(next(it))) % 256
                    else:  # 当前像素是中间的像素，其要接受前面像素扩散来的信息
                        flt[i] = (flt[i - 1] + flt[i] + utils.discrete(next(it))) % 256
            else:  # 逆向扩散，此时传入的it是已经逆向过的序列发生器
                for i in reversed(range(len(flt))):  # 逆向遍历
                    if i == 0:
                        flt[i] = (flt[i] - utils.discrete(next(it))) % 256  # 从+变-
                    else:
                        flt[i] = (flt[i] - flt[i - 1] - utils.discrete(next(it))) % 256  # 从+变-
        return flt.reshape(shape)  # 还原成二维图像
    

    def get_cost(self, rgb):
        return rgb.shape[0] * rgb.shape[1] * rgb.shape[2] * self.times


@operation_registry.register('Compositional')
class CompositionalChaosOperation(BaseOperation):  # 组合的加密操作，可以把多个操作封装成一个操作
    def __init__(self, op_list, times=1):
        self.op_list = op_list  # 要执行的操作列表
        self.times = times

    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):
            if not reverse:
                for op in self.op_list:  # 依次执行所有的操作
                    rgb = op(rgb, it, reverse)
            else:
                for op in reversed(self.op_list):
                    rgb = op(rgb, it, reverse)
        return rgb

    def get_cost(self, rgb):  # 组合操作的cost是所有子操作的cost之和
        cnt = 0
        for op in self.op_list:
            cnt += op.get_cost(rgb)
        return cnt * self.times
