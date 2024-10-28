"""
ChaMo_v2: Test cases for filtering implementations.
"""

import unittest
import numpy as np
from src.filtering.traditional_filters import (
    SavitzkyGolayFilter,
    FFTFilter,
    ButterworthFilter
)
from src.filtering.adaptive_filters import AdaptivePatternFilter

class TestTraditionalFilters(unittest.TestCase):
    def setUp(self):
        """Setup test data"""
        # Generate test signal
        t = np.linspace(0, 10, 1000)
        self.clean_signal = np.sin(2*np.pi*t)
        self.noisy_signal = self.clean_signal + 0.5*np.random.normal(size=1000)

    def test_savgol_filter(self):
        """Test Savitzky-Golay filter"""
        filter = SavitzkyGolayFilter()
        filtered = filter.filter(self.noisy_signal)
        
        # Check output shape
        self.assertEqual(len(filtered), len(self.noisy_signal))
        
        # Check noise reduction
        original_noise = np.std(self.noisy_signal - self.clean_signal)
        filtered_noise = np.std(filtered - self.clean_signal)
        self.assertLess(filtered_noise, original_noise)

    def test_fft_filter(self):
        """Test FFT filter"""
        filter = FFTFilter()
        filtered = filter.filter(self.noisy_signal)
        
        # Check output shape
        self.assertEqual(len(filtered), len(self.noisy_signal))
        
        # Check noise reduction
        original_noise = np.std(self.noisy_signal - self.clean_signal)
        filtered_noise = np.std(filtered - self.clean_signal)
        self.assertLess(filtered_noise, original_noise)

    def test_butterworth_filter(self):
        """Test Butterworth filter"""
        filter = ButterworthFilter()
        filtered = filter.filter(self.noisy_signal)
        
        # Check output shape
        self.assertEqual(len(filtered), len(self.noisy_signal))
        
        # Check noise reduction
        original_noise = np.std(self.noisy_signal - self.clean_signal)
        filtered_noise = np.std(filtered - self.clean_signal)
        self.assertLess(filtered_noise, original_noise)

class TestAdaptiveFilter(unittest.TestCase):
    def setUp(self):
        """Setup test data"""
        # Generate test signal with repeating patterns
        t = np.linspace(0, 10, 1000)
        pattern = np.sin(2*np.pi*t[:100])
        self.clean_signal = np.tile(pattern, 10)
        self.noisy_signal = self.clean_signal + 0.5*np.random.normal(size=1000)

    def test_adaptive_filter(self):
        """Test adaptive pattern filter"""
        filter = AdaptivePatternFilter()
        filtered = filter.filter(self.noisy_signal)
        
        # Check output shape
        self.assertEqual(len(filtered), len(self.noisy_signal))
        
        # Check noise reduction
        original_noise = np.std(self.noisy_signal - self.clean_signal)
        filtered_noise = np.std(filtered - self.clean_signal)
        self.assertLess(filtered_noise, original_noise)
        
        # Check pattern learning
        stats = filter.get_pattern_statistics()
        self.assertGreater(stats['total_patterns'], 0)

if __name__ == '__main__':
    unittest.main()