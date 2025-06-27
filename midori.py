import numpy as np
from math import floor

class MidoriCipher:
    def __init__(self):
        self.sbox0_table = [0xc, 0xa, 0xd, 0x3, 0xe, 0xb, 0xf, 0x7,
                            0x8, 0x9, 0x1, 0x5, 0x0, 0x2, 0x4, 0x6]
        self.sbox1_table = [0x1, 0x0, 0x5, 0x3, 0xe, 0x2, 0xf, 0x7,
                            0xd, 0xa, 0x9, 0xb, 0xc, 0x8, 0x4, 0x6]

        self.inv_sbox0_table = [0] * 16
        self.inv_sbox1_table = [0] * 16
        for i in range(16):
            self.inv_sbox0_table[self.sbox0_table[i]] = i
            self.inv_sbox1_table[self.sbox1_table[i]] = i
        self.round_constants = [
            np.array([[0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]),
            np.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]]),
            np.array([[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]]),
            np.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1], [1, 0, 0, 1]]),
            np.array([[1, 0, 0, 0], [1, 0, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]]),
            np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]),
            np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]]),
            np.array([[1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]),
            np.array([[0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]),
            np.array([[0, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]]),
            np.array([[0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]]),
            np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]]),
            np.array([[1, 1, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0]]),
            np.array([[1, 1, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 0]]),
            np.array([[0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1]]),
            np.array([[0, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]),
            np.array([[0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 0], [0, 1, 1, 0]]),
            np.array([[0, 0, 1, 1], [1, 0, 0, 0], [1, 1, 0, 1], [0, 0, 0, 0]])
        ]
        self.mix_matrix = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ])
        self.inv_mix_matrix = self.mix_matrix.copy()
        self.shuffle_permutation = [0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8]
        self.inv_shuffle_permutation = [0, 7, 14, 9, 8, 2, 11, 12, 15, 8, 1, 6, 10, 13, 4, 3]
        self._fix_inverse_permutation()

    def _fix_inverse_permutation(self):
        self.inv_shuffle_permutation = [0] * 16
        for i in range(16):
            self.inv_shuffle_permutation[self.shuffle_permutation[i]] = i

    def _sbox(self, nibble, sbox_type):
        if sbox_type == 0:
            return self.sbox0_table[nibble]
        else:
            return self.sbox1_table[nibble]

    def _inv_sbox(self, nibble, sbox_type):
        if sbox_type == 0:
            return self.inv_sbox0_table[nibble]
        else:
            return self.inv_sbox1_table[nibble]

    def _apply_sbox_to_byte(self, byte_val, sbox_idx, inverse=False):
        bits = [(byte_val >> i) & 1 for i in range(7, -1, -1)]
        if inverse:
            if sbox_idx == 0:
                n0_input = bits[4] << 3 | bits[1] << 2 | bits[6] << 1 | bits[3]
                n1_input = bits[0] << 3 | bits[5] << 2 | bits[2] << 1 | bits[7]
                n0_output = self._inv_sbox(n0_input, 1)
                n1_output = self._inv_sbox(n1_input, 1)
                result_bits = [
                    (n1_output >> 3) & 1, (n0_output >> 2) & 1, (n1_output >> 1) & 1, (n0_output >> 0) & 1,
                    (n0_output >> 3) & 1, (n1_output >> 2) & 1, (n0_output >> 1) & 1, (n1_output >> 0) & 1
                ]
            elif sbox_idx == 1:
                n0_input = bits[1] << 3 | bits[6] << 2 | bits[7] << 1 | bits[0]
                n1_input = bits[5] << 3 | bits[2] << 2 | bits[3] << 1 | bits[4]
                n0_output = self._inv_sbox(n0_input, 1)
                n1_output = self._inv_sbox(n1_input, 1)
                result_bits = [
                    (n0_output >> 0) & 1, (n0_output >> 3) & 1, (n1_output >> 2) & 1, (n1_output >> 1) & 1,
                    (n1_output >> 0) & 1, (n1_output >> 3) & 1, (n0_output >> 2) & 1, (n0_output >> 1) & 1
                ]
            elif sbox_idx == 2:
                n0_input = bits[2] << 3 | bits[3] << 2 | bits[4] << 1 | bits[1]
                n1_input = bits[6] << 3 | bits[7] << 2 | bits[0] << 1 | bits[5]
                n0_output = self._inv_sbox(n0_input, 1)
                n1_output = self._inv_sbox(n1_input, 1)
                result_bits = [
                    (n1_output >> 1) & 1, (n0_output >> 0) & 1, (n0_output >> 3) & 1, (n0_output >> 2) & 1,
                    (n0_output >> 1) & 1, (n1_output >> 0) & 1, (n1_output >> 3) & 1, (n1_output >> 2) & 1
                ]
            else:
                n0_input = bits[7] << 3 | bits[4] << 2 | bits[1] << 1 | bits[2]
                n1_input = bits[3] << 3 | bits[0] << 2 | bits[5] << 1 | bits[6]
                n0_output = self._inv_sbox(n0_input, 1)
                n1_output = self._inv_sbox(n1_input, 1)
                result_bits = [
                    (n1_output >> 2) & 1, (n0_output >> 1) & 1, (n0_output >> 0) & 1, (n1_output >> 3) & 1,
                    (n0_output >> 2) & 1, (n1_output >> 1) & 1, (n1_output >> 0) & 1, (n0_output >> 3) & 1
                ]
        else:
            if sbox_idx == 0:
                n0_input = bits[4] << 3 | bits[1] << 2 | bits[6] << 1 | bits[3]
                n1_input = bits[0] << 3 | bits[5] << 2 | bits[2] << 1 | bits[7]
                n0_output = self._sbox(n0_input, 1)
                n1_output = self._sbox(n1_input, 1)
                result_bits = [
                    (n1_output >> 3) & 1, (n0_output >> 2) & 1, (n1_output >> 1) & 1, (n0_output >> 0) & 1,
                    (n0_output >> 3) & 1, (n1_output >> 2) & 1, (n0_output >> 1) & 1, (n1_output >> 0) & 1
                ]
            elif sbox_idx == 1:
                n0_input = bits[1] << 3 | bits[6] << 2 | bits[7] << 1 | bits[0]
                n1_input = bits[5] << 3 | bits[2] << 2 | bits[3] << 1 | bits[4]
                n0_output = self._sbox(n0_input, 1)
                n1_output = self._sbox(n1_input, 1)
                result_bits = [
                    (n0_output >> 0) & 1, (n0_output >> 3) & 1, (n1_output >> 2) & 1, (n1_output >> 1) & 1,
                    (n1_output >> 0) & 1, (n1_output >> 3) & 1, (n0_output >> 2) & 1, (n0_output >> 1) & 1
                ]
            elif sbox_idx == 2:
                n0_input = bits[2] << 3 | bits[3] << 2 | bits[4] << 1 | bits[1]
                n1_input = bits[6] << 3 | bits[7] << 2 | bits[0] << 1 | bits[5]
                n0_output = self._sbox(n0_input, 1)
                n1_output = self._sbox(n1_input, 1)
                result_bits = [
                    (n1_output >> 1) & 1, (n0_output >> 0) & 1, (n0_output >> 3) & 1, (n0_output >> 2) & 1,
                    (n0_output >> 1) & 1, (n1_output >> 0) & 1, (n1_output >> 3) & 1, (n1_output >> 2) & 1
                ]
            else:
                n0_input = bits[7] << 3 | bits[4] << 2 | bits[1] << 1 | bits[2]
                n1_input = bits[3] << 3 | bits[0] << 2 | bits[5] << 1 | bits[6]
                n0_output = self._sbox(n0_input, 1)
                n1_output = self._sbox(n1_input, 1)
                result_bits = [
                    (n1_output >> 2) & 1, (n0_output >> 1) & 1, (n0_output >> 0) & 1, (n1_output >> 3) & 1,
                    (n0_output >> 2) & 1, (n1_output >> 1) & 1, (n1_output >> 0) & 1, (n0_output >> 3) & 1
                ]
        return sum(bit << (7 - i) for i, bit in enumerate(result_bits))

    def _hex_to_state(self, hex_str):
        hex_clean = hex_str.replace('0x', '').zfill(32)
        state = np.zeros((4, 4), dtype=np.uint8)
        for i in range(16):
            byte_val = int(hex_clean[i * 2:(i + 1) * 2], 16)
            row, col = i % 4, i // 4
            state[row, col] = byte_val
        return state

    def _state_to_hex(self, state):
        hex_bytes = []
        for col in range(4):
            for row in range(4):
                hex_bytes.append(f'{state[row, col]:02x}')
        return '0x' + ''.join(hex_bytes)

    def _sub_cell(self, state, inverse=False):
        for col in range(4):
            for row in range(4):
                cell_idx = col * 4 + row
                sbox_idx = cell_idx % 4
                state[row, col] = self._apply_sbox_to_byte(state[row, col], sbox_idx, inverse)
        return state

    def _shuffle_cell(self, state):
        flat_state = []
        for col in range(4):
            for row in range(4):
                flat_state.append(state[row, col])
        new_state = np.zeros((4, 4), dtype=np.uint8)
        for i in range(16):
            new_val = flat_state[self.shuffle_permutation[i]]
            row, col = i % 4, i // 4
            new_state[row, col] = new_val
        return new_state

    def _inv_shuffle_cell(self, state):
        flat_state = []
        for col in range(4):
            for row in range(4):
                flat_state.append(state[row, col])
        new_state = np.zeros((4, 4), dtype=np.uint8)
        for i in range(16):
            new_val = flat_state[self.inv_shuffle_permutation[i]]
            row, col = i % 4, i // 4
            new_state[row, col] = new_val
        return new_state

    def _mix_column(self, state, inverse=False):
        matrix = self.inv_mix_matrix if inverse else self.mix_matrix
        result = np.zeros((4, 4), dtype=np.uint8)
        for col in range(4):
            column = state[:, col].copy()
            for row in range(4):
                temp = 0
                for k in range(4):
                    if matrix[row, k] == 1:
                        temp ^= column[k]
                result[row, col] = temp
        return result

    def _key_add(self, state, key_hex):
        key_state = self._hex_to_state(key_hex)
        return state ^ key_state

    def _generate_round_keys(self, master_key_hex):
        round_keys = []
        master_key_state = self._hex_to_state(master_key_hex)
        for r in range(19):
            round_key = master_key_state ^ self.round_constants[r]
            round_keys.append(self._state_to_hex(round_key))
        return round_keys

    def encrypt(self, plaintext_hex, key_hex, rounds=20):
        state = self._hex_to_state(plaintext_hex)
        state = self._key_add(state, key_hex)
        round_keys = self._generate_round_keys(key_hex)
        for r in range(rounds - 1):
            state = self._sub_cell(state)
            state = self._shuffle_cell(state)
            state = self._mix_column(state)
            state = self._key_add(state, round_keys[r])
        state = self._sub_cell(state)
        state = self._key_add(state, key_hex)
        return self._state_to_hex(state)

    def decrypt(self, ciphertext_hex, key_hex, rounds=20):
        state = self._hex_to_state(ciphertext_hex)
        round_keys = self._generate_round_keys(key_hex)
        state = self._key_add(state, key_hex)
        state = self._sub_cell(state, inverse=True)
        for r in range(rounds - 2, -1, -1):
            state = self._key_add(state, round_keys[r])
            state = self._mix_column(state, inverse=True)
            state = self._inv_shuffle_cell(state)
            state = self._sub_cell(state, inverse=True)
        state = self._key_add(state, key_hex)
        return self._state_to_hex(state)

_cipher = MidoriCipher()

def encrypt(plaintext_hex, key_hex, rounds=20):
    return _cipher.encrypt(plaintext_hex, key_hex, rounds)

def decrypt(ciphertext_hex, key_hex, rounds=20):
    return _cipher.decrypt(ciphertext_hex, key_hex, rounds)
