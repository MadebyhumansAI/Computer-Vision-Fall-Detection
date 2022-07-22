import unittest
from train_open_pose_qat import download_and_extract
import os

class TestDownloadAndExtract(unittest.TestCase):
    def test_download_and_extract(self):
        # Test downloading and extracting a small file
        url = 'https://github.com/github/gitignore/archive/master.zip'
        download_path = './'
        extract_path = './gitignore'
        download_and_extract(url, download_path, extract_path)
        self.assertTrue(os.path.exists(extract_path))

if __name__ == '__main__':
    unittest.main()