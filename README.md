# Federated Learning for Epileptic Seizure Prediction Across Heterogeneous EEG Datasets


## :mag: Introduction
Developing accurate and generalizable epileptic seizure prediction models from electroencephalography (EEG) data across multiple clinical sites is hindered by patient privacy regulations and significant data heterogeneity (non-IID characteristics). Federated Learning (FL) offers a privacy-preserving framework for collaborative training, but standard aggregation methods like Federated Averaging (FedAvg) can be biased by dominant datasets in heterogeneous settings. This paper investigates FL for seizure prediction using a single EEG channel across four diverse public datasets (Siena, CHB-MIT, Helsinki, NCH), representing distinct patient populations (adult, pediatric,neonate) and recording conditions. We implement privacy-preserving global normalization and propose a Random Subset Aggregation strategy, where each client trains on a fixed-size random subset of its data per round, ensuring equal contribution during aggregation. Our results show
that locally trained models fail to generalize across sites, and standard weighted FedAvg yields highly skewed performance (e.g., 89.0% accuracy on CHB-MIT but only 50.8% on Helsinki and 50.6% on NCH). In contrast, Random Subset Aggregation significantly improves performance on under-represented clients (accuracy increases to 81.7% on Helsinki and 68.7% on NCH) and achieves a superior macro-average accuracy of 77.1% and pooled accuracy of 80.0% across all sites, demonstrating a
more robust and fair global model. This work highlights the potential of balanced FL approaches for building effective and generalizable seizure prediction systems in realistic, heterogeneous multi-hospital environments while respecting data privacy.

# :fire: Setup 
Please follow the following steps to create an environment for running PedSleepMAE

```
git clone https://github.com/sauravpandey123/SeizureFed.git
cd SeizureFed
conda env create -f environment.yml
conda activate SeizureFed
```

> ⚠️ **Important:**  
> All scripts in this project require command-line arguments.  
> Use the `--help` flag with any script to view its required inputs and options.



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

## 📦 1. Datasets Preprocessing

Once the datasets have been downloaded, we process them into a format suitable for model training. First, we create a `.h5` file for every `.edf` file. Then, for each dataset, we merge these individual `.h5` files into a **single patientwise `.h5` file**, where each patient's signals and seizure labels are stored under a unique key.

This consolidated format enables efficient access to all patient data within a dataset during training.

> **Note:** Once the final patientwise `.h5` files are created, the individual `.h5` files can be deleted to save storage space.

Since every dataset has a unique structure, we provide a separate preprocessing script for each one below

```bash
python3 preprocessing/preprocess_chb.py
python3 preprocessing/preprocess_helsinki.py
python3 preprocessing/preprocess_nch.py
python3 preprocessing/preprocess_siena.py
```
## 🧪 2. Non-Federated Setup

Before introducing privacy-preserving federated learning, we establish two critical baselines: **Local Training** (per-site models) and **Centralized Training** (merged data). These help us evaluate how well the model performs without federated constraints.

### Local Training (Per-Site Models)

We train one model **independently on each dataset** (CHB, Helsinki, NCH, or Siena), simulating settings where institutions do not share data. To train and then later evaluate each model on the other datasets to test generalization, run the following:

```bash
python3 non_fed_training/train_local.py
python3 non_fed_training/evaluate_local.py
```

### Centralized Training (Collaborative Model)

In this setting, we combine patient data from all four datasets and train a single model. This provides an upper-bound baseline where all data is pooled without privacy restrictions. To train this centralized model and evaluate it on each dataset, run the following:

```bash
python3 non_fed_training/train_central.py
python3 non_fed_training/evaluate_central.py
```

## 3. Federated Learning Setup

To train a seizure prediction model collaboratively across hospitals while respecting data privacy, we implement a FL setup that simulates multiple clients and a central server. Each client represents a hospital with its own private EEG dataset, and the server coordinates training without accessing raw data.

### ⚙️ Aggregation Strategies Supported

We provide three FL aggregation strategies:

- `fedavg_simple`: Standard Federated Averaging without weighting  
- `fedavg_weighted`: Weighted Federated Averaging (clients contribute proportionally to their dataset sizes)  
- `rsa`: **Random Subset Aggregation (RSA)**, where each client trains on a fixed-size subset per round to ensure balanced aggregation regardless of dataset size

Additionally, we compute **global normalization statistics** (mean and std) in a privacy-preserving way to standardize EEG features across clients.

---

> ⚠️ **Important:**  
> You **must** update the path to the `.h5` dataset files in the `Client` class before running federated training.  
>  
> Inside `client.py`, modify this line based on where your preprocessed `.h5` files are stored:
> ```python
> self.data_path = f'../DATA/{self.name.lower()}_patientwise.h5'  # <-- Update this path
> ```
> For example, if your files are in `./data/`, change it to:
> ```python
> self.data_path = f'./data/{self.name.lower()}_patientwise.h5'
> ```

---

### 🧪 Training the Federated Model

To run federated training with your desired settings:

```bash
python3 fed_training/main.py \
  --method rsa \
  --subset_size 10000 \
  --num_rounds 30 \
  --lr 4e-5 \
  --weight_decay 1e-5 \
  --local_epochs 1 \
  --channel_name "F3-C3" \
  --model_prefix "fed_rsa" \
  --model_save_dir "./saved_models" \
  --seed 42

