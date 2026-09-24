"""Step 2 - train the CEBRA hybrid embedding (manuscript Appendix E).

Reads  outputs/cebra_prep_<tag>_*.npz
Writes outputs/cebra_embeddings_<tag>_*.npz + the model weights.
~6 min on an Apple M1 Pro (MPS) or a Colab T4."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("train_cebra", run_name="__main__")
