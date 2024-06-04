import utils
import encrypt

path='img/Lenna.jpg'

rgb = utils.read_rgb(path)
arnold = encrypt.ArnoldEncryptor(a=1, b=3, shuffle_times=6)

cipher = arnold.encrypt(rgb)
utils.show_rgb(cipher)

re = arnold.decrypt(cipher)
utils.show_rgb(re)