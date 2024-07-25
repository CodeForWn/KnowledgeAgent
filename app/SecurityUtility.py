from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import binascii
import os
from Cryptos import Cryptos
from Digests import Digests


class SecurityUtility:
    key = binascii.unhexlify('642990b695eda1d321f88bdefbbdfa2e')
    iv = binascii.unhexlify('b1db978d76eabd87d75db29f6073250f')
    SALT_SIZE = 8
    HASH_ITERATIONS = 1024

    @staticmethod
    def encrypt(plain_text):
        encrypted = Cryptos.aesEncrypt(plain_text.encode('utf-8'), SecurityUtility.key, SecurityUtility.iv)
        return binascii.hexlify(encrypted).decode('utf-8')

    @staticmethod
    def decrypt(encrypted_text):
        encrypted = binascii.unhexlify(encrypted_text)
        decrypted = Cryptos.aesDecrypt(encrypted, SecurityUtility.key, SecurityUtility.iv)
        return decrypted.decode('utf-8')

    @staticmethod
    def desensitize(original, start, end):
        if not original:
            return original
        length = len(original)
        if length < (start + end):
            return original
        return original[:start] + '*' * (length - start - end) + original[-end:]

    @staticmethod
    def compare_password(plain_password, coding_password, coding_salt):
        if not plain_password:
            return False
        encoding_password = SecurityUtility.encode_password(plain_password, coding_salt)
        return coding_password == encoding_password

    @staticmethod
    def encode_password(plain_password, en_code_salt):
        salt = binascii.unhexlify(en_code_salt)
        hash_password = Digests.sha1(plain_password.encode('utf-8'), salt, SecurityUtility.HASH_ITERATIONS)
        return binascii.hexlify(hash_password).decode('utf-8')

    @staticmethod
    def generate_salt():
        return Digests.generateSalt(SecurityUtility.SALT_SIZE)

    @staticmethod
    def generate_encode_salt(salt):
        return binascii.hexlify(salt).decode('utf-8')

    @staticmethod
    def generate_password(plain_password, salt):
        hash_password = Digests.sha1(plain_password.encode('utf-8'), salt, SecurityUtility.HASH_ITERATIONS)
        return binascii.hexlify(hash_password).decode('utf-8')


# # 测试加密与解密
# plain_text = "test_id_123"
# encrypted_text = SecurityUtility.encrypt(plain_text)
# decrypted_text = SecurityUtility.decrypt(encrypted_text)
# print(f"Plain: {plain_text}, Encrypted: {encrypted_text}, Decrypted: {decrypted_text}")
#
# # 测试密码哈希与验证
# plain_password = "secure_password"
# salt = SecurityUtility.generate_salt()
# encoded_salt = SecurityUtility.generate_encode_salt(salt)
# hashed_password = SecurityUtility.generate_password(plain_password, salt)
# is_valid = SecurityUtility.compare_password(plain_password, hashed_password, encoded_salt)
# print(f"Is password valid: {is_valid}")
#
# # 测试数据脱敏
# original_text = "SensitiveInformation"
# desensitized_text = SecurityUtility.desensitize(original_text, 2, 4)
# print(f"Original: {original_text}, Desensitized: {desensitized_text}")
