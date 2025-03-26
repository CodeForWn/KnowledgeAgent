import hashlib
import os


class Digests:

    @staticmethod
    def sha1(input_data, salt, iterations):
        digest = hashlib.sha1()
        if salt is not None:
            digest.update(salt)
        digest.update(input_data)
        result = digest.digest()
        for _ in range(1, iterations):
            digest = hashlib.sha1()
            digest.update(result)
            result = digest.digest()
        return result

    @staticmethod
    def generateSalt(num_bytes):
        return os.urandom(num_bytes)

