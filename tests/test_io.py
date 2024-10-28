"""
ChaMo_v2: Test cases for I/O operations.
"""

import unittest
import numpy as np
from pathlib import Path
from src.io_utils.atf_handler import ATFHandler

class TestATFHandler(unittest.TestCase):
    def setUp(self):
        """Setup test data and create sample ATF file"""
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample ATF file
        self.test_file = self.test_dir / "test.atf"
        self._create_sample_atf()
        
        self.handler = ATFHandler(str(self.test_file))

    def tearDown(self):
        """Clean up test files"""
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def _create_sample_atf(self):
        """Create a sample ATF file for testing"""
        content = """ATF 1.0
4 2
"Time (s)" "Current (pA)"
AcquisitionMode=Episode
SampleInterval=0.0001
StartTime=0.0
0.0000  -1.2345
0.0001  -1.3456
0.0002  -1.4567
0.0003  -1.5678
0.0004  -1.6789
"""
        with open(self.test_file, 'w') as f:
            f.write(content)

    def test_load_atf(self):
        """Test loading ATF file"""
        # Test loading
        success = self.handler.load_atf()
        self.assertTrue(success)
        
        # Check data
        self.assertIsNotNone(self.handler.data)
        self.assertEqual(len(self.handler.data), 5)
        self.assertEqual(self.handler.data.shape[1], 2)

    def test_get_time_data(self):
        """Test extracting time data"""
        self.handler.load_atf()
        time_data = self.handler.get_time_data()
        
        self.assertIsNotNone(time_data)
        self.assertEqual(len(time_data), 5)
        self.assertEqual(time_data[0], 0.0)
        self.assertEqual(time_data[-1], 0.0004)

    def test_get_current_data(self):
        """Test extracting current data"""
        self.handler.load_atf()
        current_data = self.handler.get_current_data()
        
        self.assertIsNotNone(current_data)
        self.assertEqual(len(current_data), 5)
        self.assertEqual(current_data[0], -1.2345)
        self.assertEqual(current_data[-1], -1.6789)

    def test_get_sampling_rate(self):
        """Test getting sampling rate"""
        self.handler.load_atf()
        sampling_rate = self.handler.get_sampling_rate()
        
        self.assertIsNotNone(sampling_rate)
        self.assertEqual(sampling_rate, 10000.0)  # 1/0.0001

    def test_invalid_file(self):
        """Test handling invalid file"""
        # Create invalid file
        invalid_file = self.test_dir / "invalid.atf"
        invalid_file.write_text("Invalid content")
        
        handler = ATFHandler(str(invalid_file))
        success = handler.load_atf()
        
        self.assertFalse(success)
        invalid_file.unlink()

if __name__ == '__main__':
    unittest.main()