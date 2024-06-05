import utils
import encrypt

path='img/Lenna.jpg'

rgb = utils.read_rgb(path)
utils.show_rgb(rgb)
arnold = encrypt.DiscreteCosineTransform()

cipher = arnold.encrypt(rgb)
# utils.show_rgb(cipher)

re = arnold.decrypt(cipher)
utils.show_rgb(re)

print(rgb)
print(re)

