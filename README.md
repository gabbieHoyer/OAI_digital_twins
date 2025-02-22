# Foundations of a Knee Joint Digital Twin from qMRI Biomarkers for Osteoarthritis and Knee Replacement

## Overview
This repository contains code affiliated with the study, “Foundations of a Knee Joint Digital Twin from qMRI Biomarkers for Osteoarthritis and Knee Replacement.” The study forms the basis of a digital twin system of the knee joint, using advanced quantitative MRI (qMRI) and machine learning to advance precision health in osteoarthritis (OA) management and knee replacement (KR) prediction. We combined deep learning-based segmentation of knee joint structures with dimensionality reduction to create an embedded feature space of imaging biomarkers. Through cross-sectional cohort analysis and statistical modeling, we identified specific biomarkers, including variations in cartilage thickness and medial meniscus shape, that are strongly tied to OA incidence and KR outcomes. Integrating these findings into a wide-ranging framework marks a meaningful step toward personalized knee-joint digital twins, potentially strengthening therapeutic strategies and informing clinical decision-making in rheumatological care. This flexible and consistent infrastructure can be adapted to a broad range of clinical approaches in precision health.

## Complementary Codebase
For related functionality, see the companion repository:
[OAI-PC-mode-interpreter](https://github.com/gabbieHoyer/OAI-PC-mode-interpreter)

## Installation
To set up this project on your local machine, follow these steps:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/gabbieHoyer/OAI_digital_twins.git
   ```

2. **Install required Python packages**  
    ```bash
    pip install -r requirements.txt
    ```

## Usage
This codebase is divided into separate scripts for different tasks. If you prefer an interactive approach, see the example notebooks in the `notebooks` folder:

1. **Data Processing and Statistical Cohort Matching**:
  ```bash
  python scripts/twin_match.py
  ```
2. **Validate Cohort Match Quality**:
  ```bash
  python scripts/match_eval.py
  ```
3. **Feature Selection and Multivariate Regression**:
  ```bash
  python scripts/statistical_modeling.py
  ```

## Authors
Gabrielle Hoyer: (https://gabbiehoyer.github.io/)

## Publication
For more details on this work, refer to our publication:
Foundations of a knee joint digital twinfrom qMRI biomarkers for osteoarthritisand knee replacement (https://www.nature.com/articles/s41746-025-01507-3)

## Reference

```bibtex
@article{Hoyer2025Foundations,
  title  = {Foundations of a knee joint digital twin from qMRI biomarkers for osteoarthritis and knee replacement},
  author = {Hoyer, Gabrielle and Gao, K.T. and Gassert, F.G. and Luitjens, J. and Jiang, F. and Majumdar, S. and Pedoia, V.},
  journal = {npj Digit. Med.},
  volume = {8},
  number = {1},
  pages  = {1--15},
  year   = {2025},
  doi    = {10.1038/s41746-025-01507-3}
}
```