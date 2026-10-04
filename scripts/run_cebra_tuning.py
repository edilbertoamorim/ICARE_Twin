"""Optional, one-off - choose CEBRA hyperparameters by cross-validation on the
training patients (the test split is never read). See src/cebra_tuning.py.

Reads  data/dataset/PPNet_data_train.npz
Writes metrics/cebra/tuning_cv.csv, tuning_summary.csv, tuning_selected.json
Default grid: 30 points x 5 folds = 150 trainings (~12 h on an M1 Pro; resumable)."""
import runpy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

runpy.run_module("cebra_tuning", run_name="__main__")
