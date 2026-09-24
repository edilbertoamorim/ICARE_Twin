# Class encoding

Why `constants.CLASS_NAMES` is ordered the way it is.

`src/constants.py` must match what preprocessing writes. It does:

```python
CLASS_NAMES = ['BurstSupp', 'Seizure', 'LPD', 'GPD',
               'LRDA', 'GRDA', 'Continuous', 'Discontinuous']
```

Four independent confirmations:

1. **`ppnet_preprocessing.py` v0.0.6** overwrites base class 0 ("Other") in
   place: `pred_c[bci < 0.5] = 0` (Burst Suppression), `= 6` (Continuous,
   BCI >= 0.9), `= 7` (Discontinuous, 0.5-0.9).
2. **`prototype_analysis.py`** — `label_map = {0:'Burst Suppression',
   1:'Seizure', 2:'LPD', 3:'GPD', 4:'LRDA', 5:'GRDA', 6:'Continuous',
   7:'Discontinuous'}`, and `prob_idx = 0 if cls in [0,6,7] else cls`, which
   only makes sense if 0/6/7 all descend from base "Other".
3. **`model_07_06_2026.ipynb`** — `CLASS_NAMES = ['Burst Suppression',
   'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Continuous', 'Discontinuous']`.
4. **The data itself** — index 0 is 100% BCI < 0.5 (median 0.063); index 6 is
   100% BCI >= 0.9; index 7 is 100% in [0.5, 0.9).

### The bug this replaced

An earlier `_constants.py` listed `['Seizure','LPD','GPD','LRDA','GRDA',
'BurstSupp','Continuous','Discontinuous']` — it omitted Burst Suppression from
position 0, sliding indices 0-5 by one. Only Continuous and Discontinuous were
right.

| index | old label | correct | share of train |
|---|---|---|---|
| 0 | Seizure | **BurstSupp** | 26.8% |
| 1 | LPD | **Seizure** | 10.0% |
| 2 | GPD | **LPD** | 1.9% |
| 3 | LRDA | **GPD** | 12.1% |
| 4 | GRDA | **LRDA** | 1.4% |
| 5 | BurstSupp | **GRDA** | 9.3% |
| 6 | Continuous | Continuous | 22.3% |
| 7 | Discontinuous | Discontinuous | 16.1% |

The tell: the old naming implied a **26.8% seizure burden** cohort-wide.

`prototype_analysis.py`'s `custom_colors` maps the same colours to the same
*names* as the old file did, so only index->name was ever wrong. `constants.py`
now keys the palette by name for that reason.

**No numbers changed** — class ids enter CEBRA training as an integer grouping,
so the embedding, retrieval and every AUROC are unaffected. Only figure labels,
and any narrative that named a state.

Anything else built on the old `_constants.py` inherits the same bug and needs
the same check.

