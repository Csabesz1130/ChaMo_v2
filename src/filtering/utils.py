"""
ChaMo_v2: Utility functions for signal processing and filtering.
Contains shared functionality used across different filter implementations.
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Optional, Union, Dict
import matplotlib.pyplot as plt

def extract_windows(signal: np.ndarray, 
                   window_size: int, 
                   overlap: float = 0.5) -> List[np.ndarray]:
    """
    Extract overlapping windows from a signal.
    
    Args:
        signal (np.ndarray): Input signal
        window_size (int): Size of each window
        overlap (float): Overlap fraction between windows (0 to 1)
        
    Returns:
        List[np.ndarray]: List of signal windows
    """
    if not 0 <= overlap < 1:
        raise ValueError("Overlap must be between 0 and 1")
        
    step = int(window_size * (1 - overlap))
    windows = []
    
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i:i + window_size])
        
    return windows

def calculate_signal_metrics(signal: np.ndarray) -> Dict[str, float]:
    """
    Calculate various signal quality metrics.
    
    Args:
        signal (np.ndarray): Input signal
        
    Returns:
        Dict[str, float]: Dictionary of metrics
    """
    metrics = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'rms': np.sqrt(np.mean(np.square(signal))),
        'peak_to_peak': np.ptp(signal),
        'snr': calculate_snr(signal) if len(signal) > 1 else 0.0
    }
    return metrics

def calculate_snr(signal: np.ndarray) -> float:
    """
    Estimate signal-to-noise ratio using statistical methods.
    
    Args:
        signal (np.ndarray): Input signal
        
    Returns:
        float: Estimated SNR in dB
    """
    # Estimate noise using difference method
    noise = np.diff(signal)
    noise_power = np.mean(np.square(noise)) / 2
    signal_power = np.mean(np.square(signal))
    
    if noise_power == 0:
        return float('inf')
        
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def apply_window(signal: np.ndarray, 
                window_type: str = 'hanning') -> np.ndarray:
    """
    Apply a window function to the signal.
    
    Args:
        signal (np.ndarray): Input signal
        window_type (str): Type of window ('hanning', 'hamming', 'blackman')
        
    Returns:
        np.ndarray: Windowed signal
    """
    if window_type == 'hanning':
        window = np.hanning(len(signal))
    elif window_type == 'hamming':
        window = np.hamming(len(signal))
    elif window_type == 'blackman':
        window = np.blackman(len(signal))
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
        
    return signal * window

def calculate_psd(signal: np.ndarray, 
                 fs: float, 
                 nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Power Spectral Density of the signal.
    
    Args:
        signal (np.ndarray): Input signal
        fs (float): Sampling frequency in Hz
        nperseg (Optional[int]): Length of each segment
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Frequencies and PSD values
    """
    if nperseg is None:
        nperseg = min(len(signal), 256)
        
    frequencies, psd = signal.welch(signal, fs, nperseg=nperseg)
    return frequencies, psd

def detect_outliers(signal: np.ndarray, 
                   threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in the signal using z-score method.
    
    Args:
        signal (np.ndarray): Input signal
        threshold (float): Z-score threshold
        
    Returns:
        np.ndarray: Boolean mask indicating outlier positions
    """
    z_scores = np.abs(signal - np.mean(signal)) / np.std(signal)
    return z_scores > threshold

def smooth_transitions(signal: np.ndarray, 
                      window_size: int = 10) -> np.ndarray:
    """
    Smooth signal transitions using moving average.
    
    Args:
        signal (np.ndarray): Input signal
        window_size (int): Size of the smoothing window
        
    Returns:
        np.ndarray: Smoothed signal
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

def align_signals(signal1: np.ndarray, 
                 signal2: np.ndarray, 
                 max_shift: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Align two signals using cross-correlation.
    
    Args:
        signal1 (np.ndarray): First signal
        signal2 (np.ndarray): Second signal
        max_shift (Optional[int]): Maximum allowed shift
        
    Returns:
        Tuple[np.ndarray, int]: Aligned signal and shift amount
    """
    if max_shift is None:
        max_shift = len(signal1) // 2
        
    correlation = signal.correlate(signal1, signal2, mode='full')
    shift = np.argmax(correlation) - (len(signal2) - 1)
    
    if abs(shift) > max_shift:
        shift = max_shift * np.sign(shift)
        
    if shift >= 0:
        aligned = np.pad(signal2[:-shift], (shift, 0), mode='edge')
    else:
        aligned = np.pad(signal2[-shift:], (0, -shift), mode='edge')
        
    return aligned, shift

def plot_signal_comparison(original: np.ndarray, 
                         filtered: np.ndarray, 
                         title: str = "Signal Comparison"):
    """
    Plot original and filtered signals for comparison.
    
    Args:
        original (np.ndarray): Original signal
        filtered (np.ndarray): Filtered signal
        title (str): Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot signals
    ax1.plot(original, label='Original', alpha=0.7)
    ax1.plot(filtered, label='Filtered', alpha=0.7)
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)
    
    # Plot difference
    ax2.plot(filtered - original, label='Difference', color='r', alpha=0.7)
    ax2.set_title('Difference (Filtered - Original)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def calculate_correlation_matrix(windows: List[np.ndarray]) -> np.ndarray:
    """
    Calculate correlation matrix between multiple signal windows.
    
    Args:
        windows (List[np.ndarray]): List of signal windows
        
    Returns:
        np.ndarray: Correlation matrix
    """
    n_windows = len(windows)
    corr_matrix = np.zeros((n_windows, n_windows))
    
    for i in range(n_windows):
        for j in range(i, n_windows):
            corr = np.corrcoef(windows[i], windows[j])[0, 1]
            corr_matrix[i, j] = corr_matrix[j, i] = corr
            
    return corr_matrix

def normalize_signal(signal: np.ndarray, 
                    method: str = 'minmax') -> np.ndarray:
    """
    Normalize signal using various methods.
    
    Args:
        signal (np.ndarray): Input signal
        method (str): Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        np.ndarray: Normalized signal
    """
    if method == 'minmax':
        min_val = np.min(signal)
        max_val = np.max(signal)
        return (signal - min_val) / (max_val - min_val)
        
    elif method == 'zscore':
        return (signal - np.mean(signal)) / np.std(signal)
        
    elif method == 'robust':
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        return (signal - median) / (mad * 1.4826)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")