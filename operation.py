from registry import operation_registry, transform_registry
import utils


def before_perform_operation():
    def decorator(op_func):
        def wrapper(*args, **kwargs):
            print(f'performing {args[0].__class__.__name__} for {args[0].times} times')
            return op_func(*args, **kwargs)
        return wrapper
    return decorator


class BaseOperation:
    def __init__(self, times=1):
        self.times = times

    def __call__(cls, rgb, it: iter, reverse=False):
        pass

    def get_cost(cls, rgb):
        pass


@operation_registry.register('RowShuffle')
class RowShuffleOperation(BaseOperation):
    @before_perform_operation()
    def __call__(self, rgb, it: iter, reverse=False):
        for _ in range(self.times):
            for dim in (range(rgb.shape[2]) if not reverse else reversed(range(rgb.shape[2]))):
                x1 = utils.discrete(next(it)) % rgb.shape[0]
                x2 = utils.discrete(next(it)) % rgb.shape[0]
                rgb[[x1, x2], :, dim] = rgb[[x2, x1], :, dim]
        return rgb

    def get_cost(self, rgb):
        return 2 * rgb.shape[2] * self.times


@operation_registry.register('ColumnShuffle')
class ColumnShuffleOperation(BaseOperation):
    @before_perform_operation()
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
class DiffusionOperation(BaseOperation):
    @before_perform_operation()
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


@operation_registry.register('Compositional')
class CompositionalChaosOperation(BaseOperation):
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

@operation_registry.register('DiscreteCosineTransform')
class DiscreteCosineTransformOperation(BaseOperation):
    def __init__(self, times=1):
        super().__init__(times)

    @before_perform_operation()
    def __call__(self, rgb, it: iter, reverse=False):
        if not reverse:
            return transform_registry.build('DiscreteCosineTransform').forward(rgb)
        else:
            return transform_registry.build('DiscreteCosineTransform').backward(rgb)
        
    def get_cost(self, rgb):
        return 0


@operation_registry.register('FourierTransform')
class FourierTransformOperation(BaseOperation):
    def __init__(self, times=1):
        super().__init__(times)

    @before_perform_operation()
    def __call__(self, rgb, it: iter, reverse=False):
        if not reverse:
            return transform_registry.build('FourierTransform').forward(rgb)
        else:
            return transform_registry.build('FourierTransform').backward(rgb)
        
    def get_cost(self, rgb):
        return 0