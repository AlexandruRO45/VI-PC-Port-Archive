import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from coder_length_vectors import candidates


class LengthVectorsTest(unittest.TestCase):
    def test_every_family_at_eof_and_inside_longer_run(self):
        for length in range(129, 141):
            sevens = 256 - length
            vector = bytes([7] * sevens + [8] * (length - sevens))
            self.assertIn((2, length), list(candidates(b"x\x08" + vector)))
            self.assertIn((0, length), list(candidates(vector + bytes([7, 8]))))

    def test_wrong_kraft_sum(self):
        self.assertEqual(list(candidates(bytes([8]) * 145)), [])


if __name__ == "__main__":
    unittest.main()
