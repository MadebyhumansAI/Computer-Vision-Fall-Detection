import unittest
from detect_fall import calculate_descent_velocity

class TestCalculateDescentVelocity(unittest.TestCase):
    def test_calculate_descent_velocity(self):
        # Test with two consecutive frames
        yt2 = 100
        yt1 = 200
        t2 = 2
        t1 = 1
        v = calculate_descent_velocity(yt2, yt1, t2, t1)
        self.assertEqual(v, -100)

        # Test with two frames that are 0.5 seconds apart
        yt2 = 100
        yt1 = 200
        t2 = 0.5
        t1 = 0
        v = calculate_descent_velocity(yt2, yt1, t2, t1)
        self.assertEqual(v, -200)

        # Test with two frames that are 1 second apart
        yt2 = 100
        yt1 = 200
        t2 = 1
        t1 = 0
        v = calculate_descent_velocity(yt2, yt1, t2, t1)
        self.assertEqual(v, -100)

if __name__ == '__main__':
    unittest.main()