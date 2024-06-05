import utils
import encrypt
import chaos

def check_arnold():
    path='img/Lenna.jpg'

    rgb = utils.read_rgb(path)
    arnold = encrypt.ArnoldTransform()

    cipher = arnold.encrypt(rgb)
    utils.show_rgb(cipher)

    re = arnold.decrypt(cipher)
    utils.show_rgb(re)


def check_chaos():
    en = encrypt.BaseChaosTransform()
    en.add_chaos_map(chaos.ArnoldMapping(), [1.2, 2.5])
    en.add_chaos_map(chaos.TentMapping(), 0.5)
    en.add_operation(encrypt.ColumnShuffleOperation(), times=10)
    en.add_operation(encrypt.RowShuffleOperation(), times=20)
    en.add_operation(encrypt.ColumnShuffleOperation(), times=10)
    en.add_operation(encrypt.ColumnShuffleOperation(), times=10)
    en.add_operation(encrypt.DiffusionOperation(), times=4)
    path='img/Lenna.jpg'

    rgb = utils.read_rgb(path)
    cipher = en.encrypt(rgb)
    utils.show_rgb(cipher)

    re = en.decrypt(cipher)
    utils.show_rgb(re)

if __name__ == '__main__':
    check_chaos()