import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
from pathlib import Path
import json
from typing import Dict, Any, List, Tuple, Optional
from .base_filter import NoiseFilter

class AdaptivePatternFilter(NoiseFilter):
    """
    Adaptive filter that learns and recognizes signal patterns
    to separate noise from genuine signal features.
    """
    
    def __init__(self, patterns_dir: Optional[str] = None):
        super().__init__("Adaptive Pattern Filter")
        self._parameters = {
            'window_size': 1000,
            'overlap': 0.5,
            'learning_rate': 0.1,
            'correlation_threshold': 0.7,
            'max_patterns': 50
        }
        
        # Set up pattern storage
        if patterns_dir is None:
            patterns_dir = Path(__file__).parent.parent.parent / 'data' / 'noise_patterns'
        self.patterns_dir = Path(patterns_dir)
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_file = self.patterns_dir / 'learned_patterns.json'
        
        # Initialize pattern storage
        self.signal_patterns: Dict[str, Dict] = {}
        self.load_patterns()

    def filter(self, signal: np.ndarray, **kwargs) -> np.ndarray:
        """Apply adaptive pattern-based filtering."""
        self.validate_signal(signal)
        
        # Update parameters if provided
        self._parameters.update(kwargs)
        
        # Extract windows
        windows = self._extract_windows(signal)
        
        # Process signal
        filtered_signal = self._process_signal(signal, windows)
        
        # Update pattern memory
        self.save_patterns()
        
        return filtered_signal

    def _extract_windows(self, signal: np.ndarray) -> List[np.ndarray]:
        """Extract overlapping windows from signal."""
        window_size = self._parameters['window_size']
        overlap = self._parameters['overlap']
        step = int(window_size * (1 - overlap))
        
        return [signal[i:i + window_size] 
                for i in range(0, len(signal) - window_size + 1, step)]

    def _process_signal(self, signal: np.ndarray, 
                       windows: List[np.ndarray]) -> np.ndarray:
        """Process signal using learned patterns."""
        filtered_signal = np.zeros_like(signal)
        weights = np.zeros_like(signal)
        window_size = self._parameters['window_size']
        step = int(window_size * (1 - self._parameters['overlap']))
        
        for i, window in enumerate(windows):
            # Find matching patterns
            pattern_matches = self._find_matching_patterns(window)
            
            if pattern_matches:
                # Apply pattern-based filtering
                filtered_window = self._apply_patterns(window, pattern_matches)
            else:
                # Learn new pattern
                filtered_window = self._learn_new_pattern(window)
            
            # Add to output with overlap handling
            start_idx = i * step
            end_idx = start_idx + window_size
            filtered_signal[start_idx:end_idx] += filtered_window
            weights[start_idx:end_idx] += 1
        
        # Normalize overlapping regions
        weights[weights == 0] = 1
        filtered_signal /= weights
        
        return filtered_signal

    def _find_matching_patterns(self, window: np.ndarray) -> List[Dict]:
        """Find patterns matching the current window."""
        matches = []
        
        for pattern_id, pattern in self.signal_patterns.items():
            if len(pattern['signal']) == len(window):
                correlation = np.corrcoef(window, pattern['signal'])[0, 1]
                if correlation > self._parameters['correlation_threshold']:
                    matches.append({
                        'id': pattern_id,
                        'correlation': correlation,
                        'pattern': pattern
                    })
        
        return matches

    def _apply_patterns(self, window: np.ndarray, 
                       matches: List[Dict]) -> np.ndarray:
        """Apply matched patterns to filter the window."""
        filtered_window = np.zeros_like(window)
        total_weight = 0
        
        for match in matches:
            weight = match['correlation'] * match['pattern']['confidence']
            filtered_window += weight * match['pattern']['signal']
            total_weight += weight
        
        if total_weight > 0:
            filtered_window /= total_weight
        else:
            filtered_window = window
            
        return filtered_window

    def _learn_new_pattern(self, window: np.ndarray) -> np.ndarray:
        """Learn a new pattern from the window."""
        # Check if we've reached the maximum number of patterns
        if len(self.signal_patterns) >= self._parameters['max_patterns']:
            # Remove the pattern with lowest confidence
            min_conf_id = min(self.signal_patterns.items(), 
                            key=lambda x: x[1]['confidence'])[0]
            del self.signal_patterns[min_conf_id]
        
        pattern_id = f"pattern_{len(self.signal_patterns)}"
        self.signal_patterns[pattern_id] = {
            'signal': window.copy(),
            'confidence': 0.5,
            'count': 1,
            'last_updated': 0
        }
        
        return window

    def update_pattern(self, pattern_id: str, new_window: np.ndarray) -> None:
        """Update existing pattern with new observation."""
        if pattern_id in self.signal_patterns:
            pattern = self.signal_patterns[pattern_id]
            lr = self._parameters['learning_rate']
            
            # Update pattern signal
            pattern['signal'] = (1 - lr) * pattern['signal'] + lr * new_window
            
            # Update confidence and count
            pattern['confidence'] = min(1.0, pattern['confidence'] + 0.1)
            pattern['count'] += 1
            pattern['last_updated'] = 0  # Reset update counter

    def load_patterns(self) -> None:
        """Load learned patterns from file."""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                patterns_data = json.load(f)
                self.signal_patterns = {
                    k: {
                        'signal': np.array(v['signal']),
                        'confidence': float(v['confidence']),
                        'count': int(v['count']),
                        'last_updated': int(v.get('last_updated', 0))
                    }
                    for k, v in patterns_data.items()
                }

    def save_patterns(self) -> None:
        """Save learned patterns to file."""
        patterns_data = {
            k: {
                'signal': v['signal'].tolist(),
                'confidence': float(v['confidence']),
                'count': int(v['count']),
                'last_updated': int(v['last_updated'])
            }
            for k, v in self.signal_patterns.items()
        }
        
        with open(self.patterns_file, 'w') as f:
            json.dump(patterns_data, f)

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters.copy()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        for key, value in parameters.items():
            if key in self._parameters:
                self._parameters[key] = value

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get statistics about learned patterns."""
        if not self.signal_patterns:
            return {
                'total_patterns': 0,
                'average_confidence': 0.0,
                'total_observations': 0
            }
        
        return {
            'total_patterns': len(self.signal_patterns),
            'average_confidence': np.mean([p['confidence'] 
                                         for p in self.signal_patterns.values()]),
            'total_observations': sum(p['count'] 
                                    for p in self.signal_patterns.values())
        }