"""
Configuration loader with attribute and dictionary access.
"""

import os
import yaml


class Config:
    """Config wrapper supporting dot access, dict indexing, and nested traversal."""

    def __init__(self, d: dict):
        self._raw = d
        for k, v in d.items():
            str_key = str(k)
            if isinstance(v, dict):
                val = Config(v)
            elif isinstance(v, list):
                val = [Config(item) if isinstance(item, dict) else item for item in v]
            else:
                val = v
            setattr(self, str_key, val)

    def __getitem__(self, key):
        str_key = str(key)
        if hasattr(self, str_key):
            return getattr(self, str_key)
        if key in self._raw:
            return self._raw[key]
        raise KeyError(key)

    def __contains__(self, key):
        return str(key) in self.__dict__ or key in self._raw

    def get(self, key, default=None):
        str_key = str(key)
        if hasattr(self, str_key):
            return getattr(self, str_key)
        return self._raw.get(key, default)

    def __repr__(self):
        clean = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Config({clean})"

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result


def load_config(config_path: str = None) -> Config:
    """Load config YAML file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "configs",
            "config.yaml",
        )

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(raw)
