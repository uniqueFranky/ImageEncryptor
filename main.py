import utils
from registry import encryptor_registry, chaos_operation_registry, chaos_mapping_registry


def check_arnold():
    path='img/Lenna.jpg'

    rgb = utils.read_rgb(path)
    arnold = encryptor_registry.build('Arnold', shuffle_times=5, a=3, b=2)

    cipher = arnold.encrypt(rgb)
    utils.show_rgb(cipher)

    re = arnold.decrypt(cipher)
    utils.show_rgb(re)


def check_chaos():
    en = encryptor_registry.build('BaseChaos')
    en.add_chaos_map(chaos_mapping_registry.build('Arnold'), initial=[1.2, 2.5])
    en.add_chaos_map(chaos_mapping_registry.build('Tent'), initial=0.5)
    comp = chaos_operation_registry.build('Compositional', 
                                            [
                                                chaos_operation_registry.build('RowShuffle', times=5),
                                                chaos_operation_registry.build('ColumnShuffle', times=3),
                                                chaos_operation_registry.build('Diffusion', times=3)
                                            ], 
                                            times=1)
    en.add_operation(comp)
    path='img/Lenna.jpg'

    rgb = utils.read_rgb(path)
    cipher = en.encrypt(rgb)
    utils.show_rgb(cipher)

    re = en.decrypt(cipher)
    utils.show_rgb(re)


def check_classic_chaos():
    en = encryptor_registry.build('ClassicChaos', row_shuffle_times=0, tent_initial=0.8)

    path='img/Lenna.jpg'

    rgb = utils.read_rgb(path)
    cipher = en.encrypt(rgb)
    utils.show_rgb(cipher)

    re = en.decrypt(cipher)
    utils.show_rgb(re)


if __name__ == '__main__':
    check_classic_chaos()