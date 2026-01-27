"""
Configuration management for HNSCC pipeline
"""
import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """
    Configuration management with YAML support and runtime overrides.
    
    Supports nested key access via dot notation: config.get('illumination.mode')
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration with defaults and optional custom config.
        
        Args:
            config_file: Path to custom YAML configuration file
        """
        self.data = {}
        self._load_defaults()
        if config_file:
            self._load_yaml(config_file)
    
    def _load_defaults(self):
        """Load default configuration from defaults.yaml"""
        defaults_path = Path(__file__).parent / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path, 'r') as f:
                self.data = yaml.safe_load(f) or {}
    
    def _load_yaml(self, config_file: Path):
        """Load and merge custom configuration file"""
        config_file = Path(config_file)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            custom_config = yaml.safe_load(f) or {}
        
        self._deep_merge(self.data, custom_config)
    
    def _deep_merge(self, base: dict, override: dict):
        """Deep merge override into base dictionary"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., 'illumination.mode')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.
        
        Args:
            key: Dot-separated key path
            value: Value to set
        """
        keys = key.split('.')
        data = self.data
        
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}
            data = data[k]
        
        data[keys[-1]] = value
    
    def to_dict(self) -> dict:
        """Return configuration as dictionary"""
        return self.data.copy()
    
    def save(self, path: Path):
        """Save current configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False)
    
    def __repr__(self):
        return f"Config({self.data})"
