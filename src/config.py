"""
ChaMo_v2: Configuration settings
"""

import json
from pathlib import Path
from typing import Dict, Any

class Config:
    """Application configuration handler"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".chamo"
        self.config_file = self.config_dir / "config.json"
        self.default_config = {
            'window': {
                'width': 1200,
                'height': 800,
                'maximized': True
            },
            'filters': {
                'savgol': {
                    'window_length': 51,
                    'polyorder': 3
                },
                'fft': {
                    'threshold': 0.2
                },
                'butterworth': {
                    'cutoff': 0.1,
                    'order': 5
                },
                'adaptive': {
                    'window_size': 1000,
                    'overlap': 0.5,
                    'learning_rate': 0.1
                }
            },
            'plotting': {
                'line_width': 1.0,
                'grid': True,
                'dpi': 100
            },
            'io': {
                'recent_files': [],
                'max_recent': 10,
                'default_export_format': 'csv'
            }
        }
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            self.config_dir.mkdir(exist_ok=True)
            
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    return {**self.default_config, **json.load(f)}
            
            return self.default_config.copy()
            
        except Exception as e:
            print(f"Error loading config: {e}")
            return self.default_config.copy()

    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            return True
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except:
            return default

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                config = config.setdefault(k, {})
            config[keys[-1]] = value
            return True
        except:
            return False

    def add_recent_file(self, filepath: str):
        """Add file to recent files list"""
        recent = self.config['io']['recent_files']
        if filepath in recent:
            recent.remove(filepath)
        recent.insert(0, filepath)
        self.config['io']['recent_files'] = recent[:self.config['io']['max_recent']]
        self.save_config()