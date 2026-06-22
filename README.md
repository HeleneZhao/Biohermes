This repository provides analytical codes of "Proteomic Signatures of Neuropathological Alterations in Alzheimer’s Disease: Insights from the Bio-Hermes Study" # Biohermes


System requirements

Operating system tested:
- Ubuntu 24.04.4 LTS

Software:
R 4.4.1
Key dependencies:
WGCNA 1.73
clusterProfiler 4.18.2

Software:
Python 3.11.13
Key dependencies:
LightGBM 4.6.0

Outputs include:
- DAPs results
- WGCNA results
- biomarker rankings
- machine-learning model performance metrics
- ROC curves
- Mendelian randomization results

To reproduce the manuscript results:
1. Obtain access to ADDI data.
2. Run main_analysis.R.
3. Run LightGBM.py
4. Run MR.R 
