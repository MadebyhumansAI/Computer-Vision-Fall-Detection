import unittest
from pathlib import Path
from qat_training import get_model_weights_path

class TestGetModelWeightsPath(unittest.TestCase):
    def test_get_model_weights_path(self):
        # Test if cache directory and model file are created correctly
        model_name = 'resnet50'
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        model_file = cache_dir / f"{model_name}.pt"
        self.assertEqual(get_model_weights_path(model_name), model_file)

if __name__ == '__main__':
    unittest.main()