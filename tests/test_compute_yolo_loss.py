import torch
import unittest
from qat_training import compute_yolo_loss

class TestComputeYoloLoss(unittest.TestCase):
    def test_compute_yolo_loss(self):
        # Define inputs
        outputs = torch.randn(2, 13, 13, 75)
        targets = (
            torch.randn(2, 13, 13, 4),
            torch.randint(0, 2, (2, 13, 13, 1)).float(),
            torch.randint(0, 5, (2, 13, 13, 1))
        )
        anchors = [(10, 13), (16, 30), (33, 23)]
        num_classes = 4
        image_size = 416

        # Compute loss
        loss = compute_yolo_loss(outputs, targets, anchors, num_classes, image_size)

        # Check that loss is a tensor
        self.assertIsInstance(loss, torch.Tensor)

        # Check that loss has the correct shape
        self.assertEqual(loss.shape, ())

if __name__ == '__main__':
    unittest.main()