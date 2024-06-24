import utils
from registry import encryptor_registry, operation_registry, chaos_mapping_registry, attacker_registry, metric_registry
import trans, encrypt, operation, attack, evaluate
import numpy as np

def check_random(path='./img/Lenna.jpg', do_attack=True):
    # 读入图片
    rgb = utils.read_rgb(path)

    # 创建加密器
    en = encryptor_registry.build('ClassicRandom', 2024)

    # 加密
    cipher = en.encrypt(rgb)

    # 攻击
    if do_attack:
        cipher = attacker_registry.build('RowErase', times=10)(cipher)
        cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
        cipher = attacker_registry.build('PointReplace', times=10)(cipher)
        cipher = attacker_registry.build('BlockSwap', times=10, block_size=8)(cipher)

    # 显示图片
    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)


def check_chaos(path='./img/Lenna.jpg', do_attack=True):
    # 读入图片
    rgb = utils.read_rgb(path)

    # 创建加密器
    en = encryptor_registry.build('ClassicChaos')
    
    # 加密
    cipher = en.encrypt(rgb)

    # 攻击
    if do_attack:
        cipher = attacker_registry.build('RowErase', times=10)(cipher)
        cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
        cipher = attacker_registry.build('PointReplace', times=10)(cipher)
        cipher = attacker_registry.build('BlockSwap', times=10, block_size=8)(cipher)

    # 显示图片
    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

    mc = metric_registry.build('MSE')
    print(f'MSE: {mc(rgb, re)}')

    mc = metric_registry.build('PSNR')
    print(f'PSNR: {mc(rgb, re)}')

    mc = metric_registry.build('SSIM')
    print(f'SSIM: {mc(rgb, re)}')


def check_chaos_trans(path='./img/Lenna.jpg', do_attack=True):
    # 读入图片
    rgb = utils.read_rgb(path)

    # 创建加密器
    en = encryptor_registry.build('DiscreteCosineChaos')

    # 加密
    cipher = en.encrypt(rgb)

    # 攻击
    if do_attack:
        cipher = attacker_registry.build('RowErase', times=10)(cipher)
        cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
        cipher = attacker_registry.build('PointReplace', times=10)(cipher)
        cipher = attacker_registry.build('BlockSwap', times=10, block_size=8)(cipher)

    # 显示图片
    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

    mc = metric_registry.build('MSE')
    print(f'MSE: {mc(rgb, re)}')

    mc = metric_registry.build('PSNR')
    print(f'PSNR: {mc(rgb, re)}')

    mc = metric_registry.build('SSIM')
    print(f'SSIM: {mc(rgb, re)}')

def check_random_trans(path='./img/Lenna.jpg', do_attack=True):
    # 读入图片
    rgb = utils.read_rgb(path)

    # 创建随机系统
    en = encryptor_registry.build('BaseRandom', 2024)
    # 添加加密操作
    en.add_operation(operation_registry.build('FourierTransform'))
    en.add_operation(operation_registry.build('ColumnShuffle', times=3))
    en.add_operation(operation_registry.build('RowShuffle', times=3))

    # 加密
    cipher = en.encrypt(rgb)

    # 攻击
    if do_attack:
        cipher = attacker_registry.build('RowErase', times=10)(cipher)
        cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
        cipher = attacker_registry.build('PointReplace', times=10)(cipher)
        cipher = attacker_registry.build('BlockSwap', times=10, block_size=8)(cipher)

    # 显示图片
    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

    mc = metric_registry.build('MSE')
    print(f'MSE: {mc(rgb, re)}')

    mc = metric_registry.build('PSNR')
    print(f'PSNR: {mc(rgb, re)}')

    mc = metric_registry.build('SSIM')
    print(f'SSIM: {mc(rgb, re)}')

def check_arnold(path='./img/Lenna.jpg', do_attack=True):
    # 读入图片
    rgb = utils.read_rgb(path)

    # 创建加密器
    en = encryptor_registry.build('Arnold')

    # 加密
    cipher = en.encrypt(rgb)

    # 攻击
    if do_attack:
        cipher = attacker_registry.build('RowErase', times=10)(cipher)
        cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
        cipher = attacker_registry.build('PointReplace', times=10)(cipher)
        cipher = attacker_registry.build('BlockSwap', times=10, block_size=8)(cipher)

    # 显示图片
    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

    mc = metric_registry.build('MSE')
    print(f'MSE: {mc(rgb, re)}')

    mc = metric_registry.build('PSNR')
    print(f'PSNR: {mc(rgb, re)}')

    mc = metric_registry.build('SSIM')
    print(f'SSIM: {mc(rgb, re)}')


if __name__ == '__main__':
    # 设置忽略溢出警告
    np.seterr(over='ignore')

    # 报告中的“4.3.1	基于混沌系统的加密”
    # check_chaos(do_attack=True)

    # 报告中的“4.3.2	基于离散余弦变换的加密”
    # check_chaos_trans(do_attack=True)

    # 报告中的“4.3.3	基于随机系统的傅立叶变换加密”
    check_random_trans(path='./img/avatar.jpg', do_attack=True)

    # 报告中的“4.3.4	猫脸变换加密”
    # check_arnold(do_attack=False)


