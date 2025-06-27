import unittest
from midori import encrypt, decrypt

class MidoriTest(unittest.TestCase):
    def test_encrypt1(self):
        plaintext = '0x00000000000000000000000000000000'
        key = '0x00000000000000000000000000000000'
        expexted_ciphertext = '0xc055cbb95996d14902b60574d5e728d6'
        actual = encrypt(plaintext, key)
        self.assertEqual(expexted_ciphertext, actual)

    def test_decrypt1(self):
        plaintext = '0xc055cbb95996d14902b60574d5e728d6'
        key = '0x00000000000000000000000000000000'
        expexted_ciphertext = '0x00000000000000000000000000000000'
        actual = decrypt(plaintext, key)
        self.assertEqual(expexted_ciphertext, actual)

    def test_encrypt2(self):
        plaintext = '0x51084ce6e73a5ca2ec87d7babc297543'
        key = '0x687ded3b3c85b3f35b1009863e2a8cbf'
        expexted_ciphertext = '0x1e0ac4fddff71b4c1801b73ee4afc83d'
        actual = encrypt(plaintext, key)
        self.assertEqual(expexted_ciphertext, actual)

    def test_decrypt2(self):
        plaintext = '0x1e0ac4fddff71b4c1801b73ee4afc83d'
        key = '0x687ded3b3c85b3f35b1009863e2a8cbf'
        expexted_ciphertext = '0x51084ce6e73a5ca2ec87d7babc297543'
        actual = decrypt(plaintext, key)
        self.assertEqual(expexted_ciphertext, actual)

if __name__ == '__main__':
    unittest.main()
