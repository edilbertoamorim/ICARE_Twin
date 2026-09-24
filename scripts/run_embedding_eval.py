"""Optional - embedding quality: held-out InfoNCE, silhouette, an 8-class
decoder and a segment-level CPC decoder (RandomForest vs CatBoost)."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("embedding_eval", run_name="__main__")
