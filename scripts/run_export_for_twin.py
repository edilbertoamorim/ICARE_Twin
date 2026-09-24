"""Bridge step - write the CEBRA-augmented npz the Transformer twin reads.

Reads  data/PPNet_data_*.npz  +  outputs/cebra_embeddings_*
Writes outputs/'PPNet Data {Train,Test} with CEBRA COMBO V.npz'
Run after 02. Only needed if the twin is being retrained.
"""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("export_twin_input", run_name="__main__")
