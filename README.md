# Temporal Coordination Capacity Reveals Integration Without Reportability

[![Open Data](https://img.shields.io/badge/Open%20Data-blue)](https://osf.io/tvyxz/)
[![Open Materials](https://img.shields.io/badge/Open%20Materials-green)](https://osf.io/tvyxz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

## Overview

Analysis code and data for:

**"Temporal Coordination Capacity Reveals Integration Without Reportability: A Two-Stage Model of Conscious Access"**

Published in *Neuroscience of Consciousness* (Oxford Academic)

This repository contains all derived data and analysis code necessary to reproduce the findings reported in the manuscript. Primary data sources are publicly available (see [Data Access](#data-access)).

## Key Findings

### The Two-Stage Model

**Stage 1 - Conscious Moment:**
```
Conscious moment at time t  ⟺  Φ_c(t) > θ_event
```

**Stage 2 - Reportable Consciousness:**
```
Reportable  ⟺  (Φ_c(t) > θ_event) ∧ (A(t) > θ_access)
```

### The N3 Paradox

N3 deep sleep shows **higher** coordination capacity than wakefulness (Φ_c = 0.793 vs 0.684), yet produces no reportable experience. This dissociation reveals that:

- Coordination capacity constitutes the **conscious moment itself**
- A separate **access function** determines which moments become reportable
- Consciousness is not lost in N3—its products cannot enter the workspace

### Main Results

| State | Φ_c | Reportable | Interpretation |
|-------|-----|------------|----------------|
| Propofol Sedation | 0.579 | ✗ | Reduced coordination |
| Normal Wake | 0.684 | ✓ | Baseline |
| Meditators | 0.743 | ✓ | Trained elevation |
| N3 Deep Sleep | 0.793 | ✗ | Maximum coordination, no report |
| Cessation | ~0.95 | ✗ | Near-maximal, no content |

## Quick Start

```bash
# Clone repository
git clone https://github.com/[username]/Temporal-Coordination-Consciousness.git
cd Temporal-Coordination-Consciousness

# Install dependencies
pip install -r requirements.txt

# Run analysis notebook
jupyter notebook notebooks/01_Analysis_Walkthrough.ipynb
```

## Repository Structure

```
Temporal-Coordination-Consciousness/
├── README.md                      # This file
├── LICENSE                        # MIT License
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
├── CITATION.cff                   # Citation file
│
├── data/
│   ├── README_data_access.md      # How to access primary data
│   ├── processed/                 # Derived data files
│   │   ├── MESA_CrossChannel_Results.csv
│   │   ├── Cons01_complete_gradient.csv
│   │   ├── Cons01_stage_summary.csv
│   │   ├── Cons01_subject_level.csv
│   │   ├── Meditation_Summary.csv
│   │   └── TwoStageModel_Summary.csv
│   └── figures/                   # Data for figure reproduction
│       ├── Figure1_gradient_data.csv
│       ├── Figure2_dialogue_data.csv
│       └── Figure3_meditation_data.csv
│
├── src/                           # Source code
│   ├── __init__.py
│   ├── coordination_measures.py   # Core Φ_d, Φ_f, Φ_c computation
│   ├── cross_channel_analysis.py  # Inter/intra hemispheric analysis
│   └── statistical_analysis.py    # Effect sizes, tests
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_Analysis_Walkthrough.ipynb
│   └── 02_Figure_Generation.ipynb
│
└── figures/                       # Publication figures
    ├── Fig1_Gradient.pdf
    ├── Fig2_Dialogue.pdf
    └── Fig3_Meditation.pdf
```

## Theoretical Framework

### Triadic Temporal Ontology

Three irreducible temporal measures:

**Duration (Φ_d):** Past → Present coupling
```
Φ_d = MI(X_past; X_present) / H(X_present)
```
Measures how much past constrains present (efficient causality).

**Frequency (Φ_f):** Present → Future projection
```
Φ_f = TE(X_present → X_future | X_past) / H(X_future)
```
Measures information flow toward future beyond past prediction.

**Coordination Capacity (Φ_c):** Balanced synthesis
```
Φ_c = 4 × √[Φ_d(1-Φ_d) × Φ_f(1-Φ_f)]
```
Maximum (Φ_c = 1) when Φ_d = Φ_f = 0.5 (balanced integration).

### Derived Measures

**Dialogue Index (DI):** Inter-hemispheric / Intra-hemispheric coordination
```
DI = Φ_c^inter / Φ_c^intra
```

**Temporal Orientation (τ):** Balance between past and future
```
τ = Φ_f / Φ_d
```
- τ < 1: Duration-dominant (past-constrained)
- τ > 1: Frequency-dominant (future-oriented)
- τ = 1: Perfect balance

## Data Access

### Derived Data (This Repository)

All processed data necessary to reproduce tables and figures are included in `data/processed/`.

### Primary Data Sources

| Dataset | Source | Access | DOI/URL |
|---------|--------|--------|---------|
| MESA Sleep | NSRR | Registration | [sleepdata.org/datasets/mesa](https://sleepdata.org/datasets/mesa) |
| Delorme Meditation | OpenNeuro | Open | [ds001787](https://openneuro.org/datasets/ds001787) |
| Rishikesh Meditation | OpenNeuro | Open | [ds003061](https://openneuro.org/datasets/ds003061) |
| Propofol | OpenNeuro | Open | [ds003768](https://openneuro.org/datasets/ds003768) |

See `data/README_data_access.md` for detailed instructions.

## Reproducing Results

### Main Tables

```python
import pandas as pd

# Table 1: N3 vs Wake (Primary Finding)
df = pd.read_csv('data/processed/Cons01_stage_summary.csv')
print(df)

# Complete gradient
gradient = pd.read_csv('data/processed/Cons01_complete_gradient.csv')
print(gradient)
```

### Main Figures

```bash
jupyter notebook notebooks/02_Figure_Generation.ipynb
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{cabot2025temporal,
  author = {Cabot, Zayin},
  title = {Temporal Coordination Capacity Reveals Integration Without 
           Reportability: A Two-Stage Model of Conscious Access},
  journal = {Neuroscience of Consciousness},
  year = {2025},
  volume = {},
  pages = {},
  doi = {}
}
```

## The Sentence to Remember

> "Consciousness corresponds to the coordination phase itself: the transient synthesis of past and future influences quantified by Φ_c(t); what varies across states is not the presence of these conscious moments, but whether their products become globally accessible."

## License

- **Code**: MIT License (see LICENSE)
- **Data**: CC-BY 4.0

## Contact

Zayin Cabot  
Independent Researcher  
Berkeley, California, USA  
[email]

## Acknowledgments

We thank the MESA study investigators, the meditation dataset contributors (Delorme, Rishikesh), and the propofol sedation study team.
