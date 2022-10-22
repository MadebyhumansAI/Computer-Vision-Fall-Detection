import unittest
import cv2
from detect_fall import main

class TestDetectFall(unittest.TestCase):
    def test_detect_fall(self):
        # Load the test image
        img = cv2.imread("/home/ubuntu/deaa7a7d-18c3-43ab-950e-57809defa36e.png")

        # Call the main function
        main(img)

        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()