import unittest
import os
import numpy as np
from detect_fall import get_joint_positions
import cv2

class TestGetJointPositions(unittest.TestCase):
    def test_get_joint_positions(self):
        # Create a temporary directory for the test images
        test_dir = "test_images"
        os.makedirs(test_dir, exist_ok=True)

        # Create a test image with a single person
        image_path = os.path.join(test_dir, "test_image.jpg")
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[240, 320, :] = 255
        cv2.imwrite(image_path, image)

        # Call the get_joint_positions function
        keypoints = get_joint_positions(test_dir)

        # Check that the keypoints have the correct shape
        self.assertEqual(keypoints.shape, (25, 3))

        # Check that the first keypoint is at the center of the image
        self.assertAlmostEqual(keypoints[0, 0], 320, delta=5)
        self.assertAlmostEqual(keypoints[0, 1], 240, delta=5)

        # Remove the temporary directory
        shutil.rmtree(test_dir)

if __name__ == '__main__':
    unittest.main()