from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    import tomllib
except Exception:  # pragma: no cover - fallback when tomllib unavailable
    tomllib = None


@dataclass(frozen=True)
class ConfigLoader:
    repo_root: Path
    config_dir: Path

    @classmethod
    def from_env(cls) -> ConfigLoader:
        repo_root_env = os.getenv("DASHCAM_REPO_ROOT")
        if repo_root_env:
            repo_root = Path(repo_root_env).expanduser().resolve()
        else:
            repo_root = Path(__file__).resolve().parents[1]

        config_dir_env = os.getenv("DASHCAM_CONFIG_DIR")
        if config_dir_env:
            config_dir = Path(config_dir_env).expanduser().resolve()
        else:
            config_dir = repo_root / "config"
        return cls(repo_root=repo_root, config_dir=config_dir)

    def _load_toml(self, path: Path) -> Optional[Dict[str, object]]:
        if not path.exists() or tomllib is None:
            return None
        return tomllib.loads(path.read_text(encoding="utf-8"))

    def load_pipeline(self) -> Dict[str, object]:
        data = self._load_toml(self.config_dir / "pipeline.toml")
        return data if isinstance(data, dict) else {}

    def load_lanes(self) -> Dict[str, object]:
        data = self._load_toml(self.config_dir / "lanes.toml")
        if isinstance(data, dict):
            return data
        data = self._load_toml(self.config_dir / "conicals.toml")
        return data if isinstance(data, dict) else {}

    def load_scoring(self) -> Dict[str, object]:
        data = self._load_toml(self.config_dir / "scoring.toml")
        return data if isinstance(data, dict) else {}

    def load_layout(self) -> Dict[str, object]:
        data = self._load_toml(self.config_dir / "layout.toml")
        return data if isinstance(data, dict) else {}

    def load_detection_overrides(self) -> Dict[str, object]:
        data = self._load_toml(self.config_dir / "detection.toml")
        return data if isinstance(data, dict) else {}
