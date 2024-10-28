import numpy as np
from scipy import signal
from scipy.fft import fft, ifft
from .base_filter import NoiseFilter
from typing import Dict, Any, Optional

class SavitzkyGolayFilter(NoiseFilter):
    """
    Savitzky-Golay filter implementation for smooth noise reduction
    while preserving higher moments of the signal.
    """
    
    def __init__(self):
        super().__init__("Savitzky-Golay Filter")
        self._parameters = {
            'window_length': 51,
            'polyorder': 3
        }

    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Savitzky-Golay filtering."""
        self.validate_signal(signal)
        
        # Update parameters if provided
        self._parameters.update(kwargs)
        
        # Ensure window length is odd
        window_length = self._parameters['window_length']
        if window_length % 2 == 0:
            window_length += 1
            
        return signal.savgol_filter(
            signal,
            window_length=window_length,
            polyorder=self._parameters['polyorder']
        )

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._parameters:
                self._parameters[key] = value


class FFTFilter(NoiseFilter):
    """
    FFT-based noise filter that removes frequency components
    below a specified threshold.
    """
    
    def __init__(self):
        super().__init__("FFT Filter")
        self._parameters = {
            'threshold': 0.1,
            'mode': 'relative'  # 'relative' or 'absolute'
        }

    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply FFT-based filtering."""
        self.validate_signal(signal)
        
        # Update parameters if provided
        self._parameters.update(kwargs)
        
        # Compute FFT
        fft_signal = fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        
        # Apply threshold
        if self._parameters['mode'] == 'relative':
            threshold = self._parameters['threshold'] * np.max(np.abs(fft_signal))
        else:
            threshold = self._parameters['threshold']
            
        # Filter frequencies
        fft_signal[np.abs(fft_signal) < threshold] = 0
        
        # Inverse FFT
        return np.real(ifft(fft_signal))

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._parameters:
                self._parameters[key] = value


class ButterworthFilter(NoiseFilter):
    """
    Butterworth filter implementation for frequency-based
    noise reduction.
    """
    
    def __init__(self):
        super().__init__("Butterworth Filter")
        self._parameters = {
            'cutoff': 0.1,
            'order': 5,
            'fs': 1000.0,  # sampling frequency in Hz
            'btype': 'low'  # 'low', 'high', 'band'
        }

    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Butterworth filtering."""
        self.validate_signal(signal)
        
        # Update parameters if provided
        self._parameters.update(kwargs)
        
        # Calculate Nyquist frequency
        nyquist = 0.5 * self._parameters['fs']
        
        # Normalize cutoff frequency
        if isinstance(self._parameters['cutoff'], (list, tuple)):
            cutoff = [f/nyquist for f in self._parameters['cutoff']]
        else:
            cutoff = self._parameters['cutoff']/nyquist
        
        # Create filter
        b, a = signal.butter(
            self._parameters['order'],
            cutoff,
            btype=self._parameters['btype'],
            analog=False
        )
        
        # Apply filter (zero-phase filtering)
        return signal.filtfilt(b, a, signal)

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._parameters:
                self._parameters[key] = value


class MedianFilter(NoiseFilter):
    """
    Median filter implementation for spike noise removal.
    """
    
    def __init__(self):
        super().__init__("Median Filter")
        self._parameters = {
            'kernel_size': 5
        }

    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply median filtering."""
        self.validate_signal(signal)
        
        # Update parameters if provided
        self._parameters.update(kwargs)
        
        return signal.medfilt(signal, kernel_size=self._parameters['kernel_size'])

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._parameters:
                self._parameters[key] = value


class KalmanFilter(NoiseFilter):
    """
    Kalman filter implementation for real-time noise reduction.
    """
    
    def __init__(self):
        super().__init__("Kalman Filter")
        self._parameters = {
            'process_variance': 1e-5,
            'measurement_variance': 1e-2,
            'initial_estimate': 0.0
        }
        self._initialized = False

    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Kalman filtering."""
        self.validate_signal(signal)
        
        # Update parameters if provided
        self._parameters.update(kwargs)
        
        # Initialize state
        filtered_signal = np.zeros_like(signal)
        estimate = self._parameters['initial_estimate']
        error_estimate = 1.0
        
        # Process each sample
        for i, measurement in enumerate(signal):
            # Prediction
            error_estimate += self._parameters['process_variance']
            
            # Update
            kalman_gain = error_estimate / (error_estimate + self._parameters['measurement_variance'])
            estimate = estimate + kalman_gain * (measurement - estimate)
            error_estimate = (1 - kalman_gain) * error_estimate
            
            filtered_signal[i] = estimate
            
        return filtered_signal

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._parameters:
                self._parameters[key] = value