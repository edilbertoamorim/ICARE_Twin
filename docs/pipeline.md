# EEG-Twin — full pipeline

From raw EEG to the manuscript figures. Purple = models, grey = data artifacts,
blue = figures.

```mermaid
flowchart TD
    EEG["raw I-CARE EEG<br/>PhysioNet, 7 sites"] --> SPARC["SPaRCNet → ProtoPNet"]
    SPARC --> PPOUT["ICARE_protopnet_results/*.npz<br/>features · 45 activations · logits"]

    QEEG["1000_ICARE_patient_10s_94f_with_spike/<br/>94 qEEG features incl. BCI"]
    OFF["physionet_files_mapped.csv<br/>time-from-ROSC offsets"]
    CLIN["ICARE_clinical.csv<br/>age · sex · vfib · ROSC · tCA · CPC"]
    SPLIT["keaton_train/test_ids.csv<br/>patient-level split"]

    PPOUT --> AGG
    QEEG --> AGG
    OFF --> AGG
    CLIN --> AGG
    SPLIT --> AGG

    AGG["aggregation<br/>6 epochs → 300 s segments<br/>'Other' split by BCI → 8 classes"]
    AGG --> NPZ["data/dataset/<br/>PPNet_data_train.npz + PPNet_data_test.npz<br/>695 / 299 patients"]

    NPZ --> CEBRA["CEBRA hybrid<br/>time + predictions + cpc_binary + probabilities → 3-D"]
    CEBRA --> EMB["cebra_embeddings"]

    NPZ --> AUG["PPNet Data … with CEBRA COMBO V.npz"]
    EMB --> AUG

    AUG --> TWIN["Transformer twin<br/>causal, dual-head: outcome + forecast"]
    CLIN --> TWIN
    TWIN --> HAND["data/twin/handoffs/<br/>twin_step4_handoff.npz<br/>twin_matching_handoff.npz"]

    HAND --> RET["retrieval<br/>0.75·trajectory + 0.25·feature → top-k analogues"]

    EMB --> F3["CEBRA trajectory globes"]
    EMB --> F2["twin in CEBRA space"]
    RET --> F2
    EMB --> FP["prototype map"]
    HAND --> FK["twin performance figures"]
    HAND --> FH["HMM state transitions"]

    classDef model fill:#872490,stroke:#333,color:#fff
    classDef data fill:#e8eaed,stroke:#888,color:#000
    classDef fig fill:#2d75d8,stroke:#333,color:#fff
    class SPARC,CEBRA,TWIN,AGG,RET model
    class EEG,PPOUT,QEEG,OFF,CLIN,SPLIT,NPZ,EMB,AUG,HAND data
    class F3,F2,FP,FK,FH fig
```

## The three models

| Model | In | Out |
|---|---|---|
| **SPaRCNet → ProtoPNet** | raw EEG epochs | 6-class IIIC + 45 prototype activations |
| **CEBRA hybrid** | 50-d PCA of ProtoPNet features + 13 qEEG | 3-D trajectory embedding |
| **Transformer twin** | 98-d hourly summaries + 5 clinical covariates | per-block P(good), next-block forecast, 256-d hidden state |

The 6 → 8 class step is not a model: `Other` is split by the burst-suppression
index into BurstSupp (BCI < 0.5), Discontinuous (0.5–0.9), Continuous (≥ 0.9).

## Status

| Branch | State |
|---|---|
| raw EEG → ProtoPNet outputs | training code not in this repo; its outputs are the pipeline's input |
| ProtoPNet outputs → `PPNet_data_*.npz` | wired as the build-dataset step; never executed |
| `PPNet_data_*.npz` → CEBRA → figures | **verified end to end, bit-reproducible** |
| `… with CEBRA COMBO V.npz` → twin → figures | **verified on a 4-patient smoke run**; full cohort never run |

CEBRA supervises on `cpc_binary` during **training only** — the encoder is
`f(X) → 3-D`, so no label enters the forward pass at inference.
