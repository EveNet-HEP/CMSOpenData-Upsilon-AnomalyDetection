# CMSOpenData-Upsilon-AnomalyDetection

This analysis uses the same data as in [1], based on the [DoubleMuon primary dataset from RunH of 2016](https://opendata.cern.ch/record/30555).

The analysis code is mostly inherited from the original code used in [1], which is available in the [`dimuonAD` repository](https://github.com/hep-lbdl/dimuonAD).

## Procedure

### Step 0: Data Preparation

The first step is to prepare the data for analysis. This involves:

1. Downloading the relevant CMS Open Data files.
2. Processing the data to extract the necessary features for anomaly detection.

We follow the data preparation steps outlined in the original `dimuonAD` repository to stay consistent with that workflow. The processed data files are provided on [Zenodo](https://zenodo.org/records/14618719).

Prepare the GitHub repository:

```bash
git clone https://github.com/hep-lbdl/dimuonAD.git
cd dimuonAD/
```

Modify `workflow.yaml`:

```yaml
file_paths:
    working_dir: [github repository path]
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
```aiignore
 python3 01_concatenate_and_filter_data.py
```
Output files will be saved as `[data storage directory path]/compiled_data/lowmass/skimmed_data_2016H_30555_nojet`

### Step 1: Installation
After preparing the data, you can run our analysis code.
```bash
git clone https://github.com/EveNet-HEP/CMSOpenData-Upsilon-AnomalyDetection.git
cd CMSOpenData-Upsilon-AnomalyDetection
```
To run the analysis, you need to install the required Python packages. You can do this using pip:

```bash
conda create --prefix [path] python=3.12
conda activate [path]
pip3 install -r requirements.txt
```
[1] Rikab Gambhir, Radha Mastandrea, Benjamin Nachman, Jesse Thaler, *Isolating Unisolated Upsilons with Anomaly Detection in CMS Open Data*, Phys. Rev. Lett. 135, 021902 (2025). DOI: [10.1103/vvv3-5kkl.135.021902](https://doi.org/10.1103/vvv3-5kkl.135.021902)
