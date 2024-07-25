from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class Cryptos:

    @staticmethod
    def aesEncrypt(data, key, iv):
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return cipher.encrypt(pad(data, AES.block_size))

    @staticmethod
    def aesDecrypt(encrypted_data, key, iv):
        cipher = AES.new(key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(encrypted_data), AES.block_size)
