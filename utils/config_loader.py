import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_dir="config"):
        self.config_dir = Path(config_dir)
        self.system_config = self._load("system_config.yaml")
        self.model_config = self._load("model_config.yaml")

    def _load(self, filename):
        path = self.config_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing config file: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)
