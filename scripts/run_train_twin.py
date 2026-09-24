"""Step 9 - train the Transformer digital twin (the model).

Reads  outputs/'PPNet Data {Train,Test} with CEBRA COMBO V.npz'  (the twin-export step)
       data/tables/ICARE_clinical.csv
Writes outputs/twin/  handoffs, weights, fig1-7, metrics/

Expensive: 25 epochs x 10 seeds x two models, CUDA or CPU only (no MPS).
Smoke test:  CEBRA_TWIN_SMOKE=1 python scripts/run_train_twin.py
"""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("twin_model", run_name="__main__")
