"""Shared constants for cebra_pipeline plotting/analysis scripts."""

# Data encoding (matches `predictions` values 0..7)
# Class encoding, fixed by the ppnet_preprocessing.py v0.0.6.
# The base SPaRCNet/ProtoPNet softmax emits 6 classes with 0 = "Other"; that
# script then splits "Other" by the burst-suppression/continuity index:
#     BCI < 0.5 -> 0 (Burst Suppression) · 0.5-0.9 -> 7 (Discontinuous)
#     BCI >= 0.9 -> 6 (Continuous)
# Verified directly against the data: index 0 is 100% BCI < 0.5 (median 0.063),
# index 6 is 100% BCI >= 0.9, index 7 is 100% in [0.5, 0.9).
CLASS_NAMES = ['BurstSupp', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA',
               'Continuous', 'Discontinuous']
N_CLASSES = len(CLASS_NAMES)

# Display order: Seizure > GPD > LPD > LRDA > GRDA > BurstSupp > Discontinuous > Continuous
DISPLAY_ORDER = [CLASS_NAMES.index(n) for n in
                 ['Seizure', 'GPD', 'LPD', 'LRDA', 'GRDA', 'BurstSupp',
                  'Discontinuous', 'Continuous']]
DISPLAY_NAMES = [CLASS_NAMES[i] for i in DISPLAY_ORDER]

# Validated categorical palette (dataviz six-checks, light surface, ALL pairs):
#   worst normal-vision dE 20.1 (floor 15) - worst CVD dE 8.0 (target 8) - PASS.
# Keyed by NAME so a future encoding change cannot silently recolour states.
_PALETTE = {
    'Seizure':       '#c6362d',   # red
    'LPD':           '#fc7ebd',   # pink
    'GPD':           '#d89009',   # amber
    'LRDA':          '#128763',   # dark teal
    'GRDA':          '#36cf75',   # green
    'BurstSupp':     '#872490',   # purple
    'Continuous':    '#2d75d8',   # blue
    'Discontinuous': '#51bdf3',   # light blue
}
CLASS_COLORS = {i: _PALETTE[n] for i, n in enumerate(CLASS_NAMES)}
CLASS_COLOR_LIST = [CLASS_COLORS[i] for i in range(N_CLASSES)]

# Outcome (binary CPC)
OUTCOME_GOOD     = '#3AA4F3'        # rgb(58,164,243)  CPC <= 2
OUTCOME_BAD      = '#FFA219'        # rgb(255,162,25)  CPC >= 3
OUTCOME_GOOD_RGB = 'rgb(58,164,243)'
OUTCOME_BAD_RGB  = 'rgb(255,162,25)'

# Uniform bin resolution
BIN_SEC = 300

# Paths come from config.py (env-overridable).
from config import DATA, OUT, CEBRA_DIR, TWIN_HANDOFF_DIR, prep, embeddings

# re-exported so every module can do `from constants import OUT_DIR`
__all__ = ['DATA', 'OUT', 'CEBRA_DIR', 'TWIN_HANDOFF_DIR', 'prep',
           'embeddings', 'CLASS_NAMES', 'CLASS_COLORS', 'CLASS_COLOR_LIST',
           'DISPLAY_ORDER', 'DISPLAY_NAMES', 'N_CLASSES', 'BIN_SEC',
           'TWIN_TO_OURS', 'OUTCOME_GOOD', 'OUTCOME_BAD',
           'OUTCOME_GOOD_RGB', 'OUTCOME_BAD_RGB']

# The twin handoff's meta JSON lists exactly this class order, and it is
# CORRECT -- both it and this file follow ppnet_preprocessing.py. The
# handoff arrays therefore need no remapping.
TWIN_TO_OURS = [0, 1, 2, 3, 4, 5, 6, 7]   # identity: same encoding
