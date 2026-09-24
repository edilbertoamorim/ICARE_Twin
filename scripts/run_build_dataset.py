"""Step 0 (standalone) - rebuild PPNet_data_*.npz from ProtoPNet outputs.

the aggregation. NOT part of run_all.py: everything downstream starts
from the npz this writes. Configure paths in src/config.py -> PREPROCESS.

*** Never executed - its inputs are not in this folder. Expect to debug it. ***
"""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("build_dataset", run_name="__main__")
