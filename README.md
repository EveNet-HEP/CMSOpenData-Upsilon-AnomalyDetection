# CMSOpenData-Upsilon-AnomalyDetection

This analysis uses the same data as in [1], based on the [DoubleMuon primary dataset from RunH of 2016](https://opendata.cern.ch/record/30555).

The analysis code is mostly inherited from the original code used in [1], which is available in the [`dimuonAD` repository](https://github.com/hep-lbdl/dimuonAD).

## Procedure

### Step 0: Data Preparation

The first step is to prepare the data for analysis. This involves:

1. Downloading the relevant CMS Open Data files.
2. Processing the data to extract the necessary features for anomaly detection.

We follow the data preparation steps outlined in the original `dimuonAD` repository to stay consistent with that workflow. The processed data files are provided on [Zenodo](https://zenodo.org/records/14618719).

At this stage, you only need the upstream `dimuonAD` repository. This repository is not required until Step 1.

Prepare the upstream repository:

```bash
git clone https://github.com/hep-lbdl/dimuonAD.git
cd dimuonAD/
```

Modify `workflow.yaml` in `dimuonAD`:

```yaml
file_paths:
    working_dir: [dimuonAD repository path]
    data_storage_dir: [data storage directory path]
```

Download the data files from the Zenodo link above and place them in the specified `data_storage_dir`:

```bash
mkdir -p [data storage directory path]/precompiled_data/skimmed_data_2016H_30555/
cd [data storage directory path]/precompiled_data/skimmed_data_2016H_30555/
curl -L --fail --retry 5 --retry-delay 5 \
  -H 'User-Agent: zenodo-downloader/1.0' \
  -OJ 'https://zenodo.org/api/records/14618719/files-archive'
unzip *.zip
```

After that, run the `dimuonAD` workflow using the paths you set in `workflow.yaml`.

If you need the environment and package setup first, complete Step 1 and then return to this step.

The output files from the `dimuonAD` preprocessing step will be saved under:

```text
[data storage path]/compiled_data/lowmass/skimmed_data_2016H_30555_nojet
```

### Step 1: Installation
After preparing the data, install this repository and set up the environment for the analysis code.
```bash
git clone https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection.git
cd CMSOpenData-Upsilon-AnomalyDetection
git clone --recursive https://github.com/EveNet-HEP/EveNet-Full.git
```
After cloning this repository, update `config/data_config.yaml` to use the same `working_dir` and `data_storage_dir` paths as in the upstream `dimuonAD` `workflow.yaml`.

Set up the environment path:
```bash
source src.sh
```
To run the analysis, install the required Python packages:
```bash
conda create --prefix [path] python=3.12
conda activate [path]
pip3 install -r requirements.txt
```

### Step 2: Data Reformatting
The next step is to reformat the data into a format suitable for our analysis.
Before running, set up the path in `config/data_config.yaml` to be our path
```yaml
file_paths:
    working_dir: [dimuonAD repository path]
    data_storage_dir: [data storage directory path]
```
And then `config/workflow.yaml` to be
```yaml
output:
  plotdir: [minor plot path]
  storedir: [results path]
```
### Step 3: Anomaly Detection Analysis
#### 3.1 Generate script
As the analysis consist of several boostrap & k-fold training. We use `Make_Script.py` to generate the pipeline in the `[farm]` directory. The command is:
```aiignore
# Quick run for testing
python3 Make_Script.py config/workflow.yaml --boostrap 2 --farm Farm-pretrain --ray_dir /pscratch/sd/t/tihsu/tmp/ --gen_events 5000 --gpu 1 --k 2 --max_background 2000 --no_signal --test_no_signal --num_toys 5 --calibrated --drop pc-log_pt-0 pc-log_pt-1 pc-log_energy-0 pc-log_energy-1 pt-balance-pc deltaR-pc pc-phi-0 pc-phi-1
```
#### 3.2 Data Preparation
The script `[farm]/prepare.sh` would perform dataset preparation for different boostrap.
```aiignore
sh prepare.sh
```
##### 3.2 Details
The function `prepare.sh` acutally runs the code:
```bash
python3 00prepare_CMSOpenData.py [workflow yaml]
```

The processed outputs will be written under:

```text
OS: [results path]/[tag]-result/[SR|SB]/data.parquet
SS: [results path]/[tag]-result_no_signal/[SR|SB]/data.parquet
```


## References
[1] Rikab Gambhir, Radha Mastandrea, Benjamin Nachman, Jesse Thaler, *Isolating Unisolated Upsilons with Anomaly Detection in CMS Open Data*, Phys. Rev. Lett. 135, 021902 (2025). DOI: [10.1103/vvv3-5kkl.135.021902](https://doi.org/10.1103/vvv3-5kkl.135.021902)
