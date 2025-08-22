from pathlib import Path
from pydantic import BaseModel
import os, yaml

class AppConfig(BaseModel):
    env: str = "dev"

def load_yaml(path: str | Path) -> dict:
    path = Path(path); 
    return yaml.safe_load(path.read_text()) if path.exists() else {}
