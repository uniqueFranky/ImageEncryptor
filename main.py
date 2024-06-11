import utils
from registry import encryptor_registry, operation_registry, chaos_mapping_registry, transform_registry, attacker_registry
import trans, encrypt, operation, attack
import numpy as np

def check_random():
    path = './img/Lenna.jpg'
    rgb = utils.read_rgb(path)

    en = encryptor_registry.build('ClassicRandom', 2024)
    cipher = en.encrypt(rgb)

    cipher = attacker_registry.build('RowErase', times=10)(cipher)
    cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
    cipher = attacker_registry.build('PointReplace', times=10)(cipher)

    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

def check_chaos():
    path = './img/Lenna.jpg'
    rgb = utils.read_rgb(path)

    en = encryptor_registry.build('ClassicChaos')
    cipher = en.encrypt(rgb)

    cipher = attacker_registry.build('RowErase', times=10)(cipher)
    cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
    cipher = attacker_registry.build('PointReplace', times=10)(cipher)

    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

def check_chaos_trans():
    path = './img/Lenna.jpg'
    rgb = utils.read_rgb(path)

    en = encryptor_registry.build('DiscreteCosineChaos')

    cipher = en.encrypt(rgb)

    cipher = attacker_registry.build('RowErase', times=10)(cipher)
    cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
    cipher = attacker_registry.build('PointReplace', times=10)(cipher)

    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

def check_random_trans():
    path = './img/Lenna.jpg'
    rgb = utils.read_rgb(path)

    en = encryptor_registry.build('BaseRandom', 2024)
    en.add_operation(operation_registry.build('FourierTransform'))
    en.add_operation(operation_registry.build('ColumnShuffle', times=3))
    en.add_operation(operation_registry.build('RowShuffle', times=3))

    cipher = en.encrypt(rgb)

    cipher = attacker_registry.build('RowErase', times=10)(cipher)
    cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
    cipher = attacker_registry.build('PointReplace', times=10)(cipher)

    utils.show_rgb(cipher)
    re = en.decrypt(cipher)
    utils.show_rgb(re)

def check_cos():
    path = './img/Lenna.jpg'
    rgb = utils.read_rgb(path)

    t = transform_registry.build('DiscreteCosineTransform')
    cipher = t.forward(rgb)

    cipher = attacker_registry.build('RowErase', times=10)(cipher)
    cipher = attacker_registry.build('ColumnErase', times=10)(cipher)
    cipher = attacker_registry.build('PointReplace', times=10)(cipher)

    utils.show_rgb(cipher)

    re = t.backward(cipher)
    utils.show_rgb(re)

if __name__ == '__main__':
    check_cos()


