# Primary Data Access Instructions

This document describes how to access the primary datasets used in this study. All primary data are publicly available through established repositories.

## Overview

| Dataset | Sample | Source | Access Type |
|---------|--------|--------|-------------|
| MESA Sleep Study | n=98 (full), n=10 (cross-channel) | NSRR | Registration |
| Delorme Meditation | n=24 (12 expert, 12 novice) | OpenNeuro | Open |
| Rishikesh Meditation | n=13 experienced | OpenNeuro | Open |
| Propofol Sedation | n=16 paired | OpenNeuro | Open |
| Cessation (Shinozuka) | n=5 advanced | In preparation | Contact |

---

## 1. MESA Sleep Study (Primary Dataset)

### Source
National Sleep Research Resource (NSRR)  
Multi-Ethnic Study of Atherosclerosis (MESA) Sleep Study

### URL
https://sleepdata.org/datasets/mesa

### Access Procedure
1. Create free account at sleepdata.org
2. Navigate to MESA dataset
3. Complete Data Use Agreement
4. Request access (typically approved within 24-48 hours)
5. Download polysomnography (PSG) files

### Files Used
- EDF files containing overnight polysomnography
- Channels: C3, C4, F3, F4, O1, O2 (referenced to contralateral mastoids)
- Sleep staging annotations (XML format)

### Sample Details

**Full Sample (Single-Channel Analysis):**
- n = 98 subjects
- 122,913 epochs (30-second windows)
- All sleep stages: Wake, N1, N2, N3, REM

**Cross-Channel Subset:**
- n = 10 subjects
- Inclusion: Complete bilateral EEG, artifact-free across all stages
- Used for: Φ_c computation, hemispheric dialogue analysis

### Preprocessing
- Epoch length: 30 seconds
- Sampling rate: 256 Hz (downsampled if necessary)
- Filtering: 0.5-45 Hz bandpass
- Artifact rejection: Epochs with amplitude > 200 µV excluded

### Citation
```
Chen X, Wang R, Zee P, et al. (2015) Racial/ethnic differences in sleep 
disturbances: The Multi-Ethnic Study of Atherosclerosis (MESA). 
Sleep 38:877-888.
```

---

## 2. Delorme Meditation Dataset

### Source
OpenNeuro

### Dataset ID
ds001787

### URL
https://openneuro.org/datasets/ds001787

### Access Procedure
1. Navigate to OpenNeuro URL
2. Download directly (no registration required)
3. Data in BIDS format

### Sample
- n = 24 participants
- 12 expert meditators (>1000 hours practice)
- 12 novice meditators (<100 hours practice)
- EEG recorded during meditation and rest

### Citation
```
Delorme A, et al. (2019) An EEG dataset recorded during eyes-closed 
resting state and meditation. OpenNeuro.
```

---

## 3. Rishikesh Meditation Dataset

### Source
OpenNeuro

### Dataset ID
ds003061

### URL
https://openneuro.org/datasets/ds003061

### Access Procedure
1. Navigate to OpenNeuro URL
2. Download directly (no registration required)
3. Data in BIDS format

### Sample
- n = 13 experienced meditators
- Recorded in Rishikesh, India
- Various meditation traditions

### Citation
```
[Original citation for ds003061]
```

---

## 4. Propofol Sedation Dataset

### Source
OpenNeuro

### Dataset ID
ds003768

### URL
https://openneuro.org/datasets/ds003768

### Access Procedure
1. Navigate to OpenNeuro URL
2. Download directly
3. Data in BIDS format

### Sample
- n = 16 participants with paired recordings
- Pre-sedation (awake) and during propofol sedation
- Standardized sedation protocol

### Preprocessing
- Same pipeline as MESA data
- Additional artifact rejection for propofol-induced changes

### Citation
```
[Original citation for ds003768]
```

---

## 5. Cessation Dataset (Preliminary)

### Source
Shinozuka et al. (in preparation)

### Access
Contact corresponding author for access
This dataset is not yet publicly available

### Sample
- n = 5 advanced meditators (>10,000 hours practice)
- Extended cessation events recorded
- **EXPLORATORY** - small sample caveat emphasized in paper

### Note
Results from this dataset are presented as preliminary/exploratory.
Replication with larger sample is needed.

---

## Derived Data

All derived data (coordination measures, summary statistics, subject-level results) are provided in the `processed/` subdirectory. These derived datasets are sufficient to reproduce all tables and figures in the manuscript without requiring access to primary data.

### Files Provided
- `MESA_CrossChannel_Results.csv`: Epoch-level coordination for cross-channel subset
- `Cons01_complete_gradient.csv`: All states gradient data
- `Cons01_stage_summary.csv`: Sleep stage summary statistics
- `Cons01_subject_level.csv`: Subject-level aggregated data
- `Meditation_Summary.csv`: Meditation dataset results
- `TwoStageModel_Summary.csv`: Two-stage model summary

---

## Processing Pipeline

For researchers who wish to reproduce the analysis from primary data:

1. **Download primary data** following instructions above
2. **Preprocess EEG**: See `src/preprocessing.py` for pipeline
3. **Compute coordination measures**: See `src/coordination_measures.py`
4. **Cross-channel analysis**: See `src/cross_channel_analysis.py`
5. **Run statistical analysis**: See `notebooks/01_Analysis_Walkthrough.ipynb`

---

## Questions

For questions about data access or processing, please open an issue on GitHub or contact the corresponding author.
