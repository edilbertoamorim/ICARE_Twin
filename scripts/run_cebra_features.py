"""Step 1 - build the CEBRA input matrix from the PPNet exports.

Reads  data/PPNet_data_{train,test}.npz
Writes outputs/cebra_prep_<tag>_{train,test}.npz + the PCA/scaler pickle."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("preprocess", run_name="__main__")
