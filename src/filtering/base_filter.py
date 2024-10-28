from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class NoiseFilter(ABC):
    """
    Abstract base class for all noise filtering implementations.
    Defines the interface that all filter types must implement.
    """

    def __init__(self, name: str):
        self.name = name
        self._parameters: Dict[str, Any] = {}
        self._is_trained = False

    @abstractmethod
    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply the filter to the input signal.
        
        Args:
            signal (np.ndarray): Input signal to be filtered
            **kwargs: Additional filter-specific parameters
            
        Returns:
            np.ndarray: Filtered signal
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current filter parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of parameter names and values
        """
        pass

    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set filter parameters.
        
        Args:
            parameters (Dict[str, Any]): Dictionary of parameter names and values
        """
        pass

    def validate_signal(self, signal: np.ndarray) -> bool:
        """
        Validate input signal format and properties.
        
        Args:
            signal (np.ndarray): Signal to validate
            
        Returns:
            bool: True if signal is valid
            
        Raises:
            ValueError: If signal is invalid
        """
        if not isinstance(signal, np.ndarray):
            raise ValueError("Signal must be a numpy array")
        
        if signal.size == 0:
            raise ValueError("Signal cannot be empty")
            
        if not np.issubdtype(signal.dtype, np.number):
            raise ValueError("Signal must contain numerical values")
            
        return True

    def get_name(self) -> str:
        """Get filter name."""
        return self.name

    def is_trained(self) -> bool:
        """Check if filter has been trained (if applicable)."""
        return self._is_trained

    def reset(self) -> None:
        """Reset filter to initial state."""
        self._is_trained = False

    def get_config(self) -> Dict[str, Any]:
        """
        Get filter configuration for serialization.
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        return {
            'name': self.name,
            'parameters': self.get_parameters(),
            'is_trained': self._is_trained
        }

    def set_config(self, config: Dict[str, Any]) -> None:
        """
        Set filter configuration from dictionary.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary
        """
        if 'parameters' in config:
            self.set_parameters(config['parameters'])
        if 'is_trained' in config:
            self._is_trained = config['is_trained']