import unittest
from detect_fall import is_fall

class TestIsFall(unittest.TestCase):
    def test_is_fall(self):
        # Test with a velocity greater than the critical speed
        v = 0.01
        self.assertTrue(is_fall(v))

        # Test with a velocity equal to the critical speed
        v = 0.009
        self.assertTrue(is_fall(v))

        # Test with a velocity less than the critical speed
        v = 0.008
        self.assertFalse(is_fall(v))

if __name__ == '__main__':
    unittest.main()