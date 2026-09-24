"""
export_twin_input.py — bridge from this pipeline to the Transformer twin.

Writes `PPNet Data {Train,Test} with CEBRA COMBO V.npz`: the original npz
with the trained CEBRA embedding appended as `cebra_embedding`.

The embedding is computed on rows sorted by (patient_id, time); the twin reads
the npz in its ORIGINAL row order, so the sort is inverted before writing.
Getting that backwards silently misaligns every segment.
"""
import numpy as np

from config import PPNET_TRAIN, PPNET_TEST, embeddings, TWIN_NPZ
from progress import step, done, log

if __name__ == '__main__':
    log('export for twin')
    for split in ('train', 'test'):
        src = PPNET_TRAIN if split == 'train' else PPNET_TEST
        step(f'{split}: reading {src.name}')
        d = np.load(src, allow_pickle=True)
        emb = np.load(embeddings(split))['embedding']

        order = np.lexsort((d['times'].astype(float), d['patient_ids']))
        if len(order) != len(emb):
            raise SystemExit(f'{split}: {len(order)} rows but {len(emb)} embeddings')
        inv = np.argsort(order)
        emb_original_order = emb[inv]

        # round-trip check: re-sorting must recover the embedding exactly
        assert np.array_equal(emb_original_order[order], emb), 'row-order inversion failed'

        out = {k: d[k] for k in d.files}
        out['cebra_embedding'] = emb_original_order
        dst = TWIN_NPZ[split]
        np.savez(dst, **out)
        done(f'{split}: {emb_original_order.shape[0]:,} rows, {len(out)} arrays', dst)
