# Federated Learning for Epileptic Seizure Prediction Across Heterogeneous EEG Datasets


## :mag: Introduction
Developing accurate and generalizable epileptic seizure prediction models from electroencephalography (EEG) data across multiple clinical sites is hindered by patient privacy regulations and significant data heterogeneity (non-IID characteristics). Federated Learning (FL) offers a privacy-preserving framework for collaborative training, but standard aggregation methods like Federated Averaging (FedAvg) can be biased by dominant datasets in heterogeneous settings. This paper investigates FL for seizure prediction using a single EEG channel across four diverse public datasets (Siena, CHB-MIT, Helsinki, NCH), representing distinct patient populations (adult, pediatric,neonate) and recording conditions. We implement privacy-preserving global normalization and propose a Random Subset Aggregation strategy, where each client trains on a fixed-size random subset of its data per round, ensuring equal contribution during aggregation. Our results show
that locally trained models fail to generalize across sites, and standard weighted FedAvg yields highly skewed performance (e.g., 89.0% accuracy on CHB-MIT but only 50.8% on Helsinki and 50.6% on NCH). In contrast, Random Subset Aggregation significantly improves performance on under-represented clients (accuracy increases to 81.7% on Helsinki and 68.7% on NCH) and achieves a superior macro-average accuracy of 77.1% and pooled accuracy of 80.0% across all sites, demonstrating a
more robust and fair global model. This work highlights the potential of balanced FL approaches for building effective and generalizable seizure prediction systems in realistic, heterogeneous multi-hospital environments while respecting data privacy.

# :fire: Setup 
Please follow the following steps to create an environment for running PedSleepMAE

```
git clone https://github.com/sauravpandey123/PedSleepMAE.git
cd PedSleepMAE
conda env create -f environment.yml
conda activate pedsleep_env
```


# :computer: User Guide 
## :inbox_tray:  0. Downloading the Datasets

We use the following publicly available EEG datasets in this project:

- [Siena Scalp EEG](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
- [Helsinki University Hospital TUEG Dataset](https://zenodo.org/records/2547147)
- [NCH Sleep Databank](https://sleepdata.org/datasets/nchsdb)

Each dataset can be downloaded using the respective links above. Please review and comply with the data usage licenses provided by the original sources.

**For the NCH Sleep Databank, only a small subset of patients experienced seizures.**  
You should download only the `.edf`, `.annot`, and `.tsv` files corresponding to these patients, as listed [here](NCH_Seizure_Patients.md).
