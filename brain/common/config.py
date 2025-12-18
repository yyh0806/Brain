"""
Configuration Module for Brain System
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration management class for Brain system
    """

    def __init__(self):
        self.config_data: Dict[str, Any] = {}
        self.default_config = {
            "sensors": {
                "lidar": {
                    "enabled": True,
                    "update_rate": 10.0
                },
                "camera": {
                    "enabled": True,
                    "update_rate": 30.0
                },
                "imu": {
                    "enabled": True,
                    "update_rate": 100.0
                },
                "gps": {
                    "enabled": True,
                    "update_rate": 10.0
                }
            },
            "world_model": {
                "grid_resolution": 0.1,
                "update_rate": 10.0,
                "max_range": 100.0
            },
            "testing": {
                "simulation_mode": True,
                "debug_mode": True
            }
        }

    def load_from_file(self, config_path: str):
        """Load configuration from file"""
        path = Path(config_path)

        if not path.exists():
            print(f"Warning: Config file {config_path} not found, using defaults")
            self.config_data = self.default_config
            return

        try:
            if path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    self.config_data = json.load(f)
            elif path.suffix.lower() in ['.yaml', '.yml']:
                with open(path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

            # Merge with defaults
            self.config_data = {**self.default_config, **self.config_data}

        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            self.config_data = self.default_config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """Set configuration value by key"""
        keys = key.split('.')
        config = self.config_data

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        self.set(key, value)