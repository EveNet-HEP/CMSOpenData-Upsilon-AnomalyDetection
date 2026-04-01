# CMSOpenData-Upsilon-AnomalyDetection

This analysis uses the same data as in [1], based on the [DoubleMuon primary dataset from RunH of 2016](https://opendata.cern.ch/record/30555).

The analysis code is mostly inherited from the original code used in [1], which is available in the [`dimuonAD` repository](https://github.com/hep-lbdl/dimuonAD).

## Quick Start

If you only want a quick run, here is the quick-start:

1. Complete [**Step 0**](###Step-0:-Data-Preparation) to prepare the data.
2. Clone this repository and install the environment in [**Step 1**](###Step-1:-Installation).
3. Update the paths and model configuration in **Step 2**.
4. Generate a small test pipeline:


```bash
python3 Make_Script.py config/workflow.yaml --boostrap 1 --farm Farm-pretrain --ray_dir [tmp dir] --gen_events 50000 --gpu 1 --k 2 --max_background 2000 --no_signal --test_no_signal --total-gpu 4 --num_toys 5 --calibrated --drop pc-log_pt-0 pc-log_pt-1 pc-log_energy-0 pc-log_energy-1 pt-balance-pc deltaR-pc pc-phi-0 pc-phi-1
```

5. Run the generated preparation script:

```bash
sh Farm-pretrain/prepare.sh
```

6. For a local smoke test, run one of the generated training scripts directly:

```bash
sh Farm-pretrain-gen/boostrap-0/train.sh
```

This is enough to check that the data reformatting, config generation, and training entry points are wired correctly. The full workflow is still somewhat environment-dependent because it assumes a Slurm/Shifter-style setup for the multi-node stages.

## Procedure

### Step 0: Data Preparation

The first step is to prepare the data for analysis. This involves:

1. Downloading the relevant CMS Open Data files.
2. Processing the data to extract the necessary features for anomaly detection.

We follow the data preparation steps outlined in the original `dimuonAD` repository to stay consistent with that workflow. The processed data files are provided on [Zenodo](https://zenodo.org/records/14618719).

At this stage, you only need the upstream `dimuonAD` repository. This repository is not required until Step 1. But if you need to set up the environment and package dependencies first, you can also use `requirements.txt` in our repository.

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
```aiignore
python3 01_concatenate_and_filter_data.py -run_samesign
```

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
# better to install zfit independently to avoid potential version conflicts with other packages
pip3 install zfit==0.22.0 # Ignore ERROR: pip's dependency...
```

Pull the Docker image for the EveNet training:
```bash
shifterimg -v pull docker:avencast1994/evenet:1.5
```
To download the pre-trained models from [Hugging Face Hub](https://huggingface.co/Avencast/EveNet), you can use the following command (`local-dir` could be replaced with any path you want)
```aiignore
hf download Avencast/EveNet --local-dir pretrain-weights
# nominal ckpt: checkpoints.20M.a4.last.ckpt
# SSL ckpt: SSL.20M.last.ckpt
```
### Step 2: Configuration Settings

Before running, update the paths in `config/data_config.yaml`:

```yaml
file_paths:
    working_dir: [dimuonAD repository path]
    data_storage_dir: [data storage directory path]
```

Then update `config/workflow.yaml`:

```yaml
output:
  plotdir: [plot path]
  storedir: [results path]
image: 'docker:avencast1994/evenet:1.5'
```

Then configure `config/full_train.yaml`:

```yaml
options:
  Training:
    pretrain_model_load_path: [your downloaded model path]
```

Most of the paths are temporary and will be overwritten by the scripts in the next steps, but make sure to set `pretrain_model_load_path` to the released model you want to use. Also remember to update `config/options.yaml` with the correct learning rate.

#### W&B Setting
Update the `src.sh` W&B API key. We recommend registering a W&B account and setting an API key to track the training process. You can set the API key in `src.sh` or export it in your terminal:

```bash
export WANDB_API_KEY=[your wandb api key]
```

In `config/full_train.yaml`, you can also set the W&B project and entity name, as well as the tags for this run. The logger name will be overwritten by the script to include the tag and bootstrap information, but you can set a base name here:
```yaml
logger:
  wandb:
    project: [your project]
    entity: [your entity]
    name: [will be overwritten by the script]
    run_name: &logger_name "nominal-test"
    tags: [ "Pretrain", "NERSC", "Multi-Node", "2700M", "SmallModel", "Vanilla", "test" ]
    simplified: false
  local:
    save_dir: [log dir] # not so important
    name: *logger_name
    version: "test-2"
```
### Step 3: Anomaly Detection Analysis

#### 3.1 Generate scripts

Since the analysis consists of several bootstrap and k-fold training steps, we use `Make_Script.py` to generate the pipeline under the `[farm]` directory. The command is:

```bash
# Quick run for testing
python3 Make_Script.py config/workflow.yaml --boostrap 1 --farm Farm-pretrain --ray_dir [tmp dir] --gen_events 50000 --gpu 1 --k 2 --max_background 2000 --no_signal --test_no_signal --total-gpu 4 --num_toys 5 --calibrated --drop pc-log_pt-0 pc-log_pt-1 pc-log_energy-0 pc-log_energy-1 pt-balance-pc deltaR-pc pc-phi-0 pc-phi-1
```

#### 3.2 Data preparation

The script `[farm]/prepare.sh` performs dataset preparation for different bootstrap runs:

```bash
sh prepare.sh
```

##### Details

The `prepare.sh` script actually runs:

```bash
python3 00prepare_CMSOpenData.py [workflow yaml]
```

The processed outputs will be written under:

```text
OS: [results path]/[tag]-result/[SR|SB]/data.parquet
SS: [results path]/[tag]-result_no_signal/[SR|SB]/data.parquet
```

`03kfold_train.py` then creates the k-fold training configuration files under the farm directory, which are used in the next step:

```bash
python3 03kfold_train.py [workflow.yaml] --fold 2 --ray_dir [tmp dir] --farm [farm/tag] --local
```

The relevant farm directories, i.e. `[farm]-gen` and `[farm]-gen-nosignal`, will also be created.

#### 3.3 Train the generative model

The training script is located at `[farm]/train.sh`. Run the following command to start the training:
##### Full Training [Optional, on slurm]
```bash
sh [farm]/train.sh
```

Please note that `[farm]/train.sh` uses `script/submit_multiple_ray.sh` to perform parallel training. If that does not work on your machine, you can instead run the entries in `train_sources` inside `farm/train.sh` iteratively on your local setup. They will look like:
##### Local mini-run
```bash
sh Farm-pretrain-gen/boostrap-0/train.sh
sh Farm-pretrain-gen-nosignal/boostrap-0/train.sh
```
The command will look like:
```aiignore
cd /global/u1/t/tihsu/CMSOpenData-Upsilon-AnomalyDetection/EveNet-Full;
shifter --image=docker:avencast1994/evenet:1.5 python3 scripts/train.py [training.yaml] --load_all --ray_dir [tmp dir]
```

#### 3.4 Generate pseudo-data
##### Full Training [Optional, on slurm]
Use parallel scripts to generate pseudo-data.
```bash
sh [farm]/run-predict.sh
```
##### Local mini-run
```bash
sh [farm]/predict.sh
```
This is typically run the signal region (SR) pseudo-data generation. The command will look like:
```aiignore
python3 01predict_SR_certain_fold.py [workflow.yaml] --region SR --checkpoint [check point]  --gen_num_events [nEvent] --ngpu 1 --rank 0 --fold 0
```

#### 3.5 Prepare the classification dataset for the weak supervision classifier training
##### Full Training / Local run
```bash
sh [farm]/train_cls_prepare.sh
```
It is typically run the signal region (SR) dataset preparation. The command will look like:
```bash
python3 02prepare_classification_dataset.py [workflow.yaml] --region SR --max_background 10000 --calibrated
python3 02prepare_classification_dataset.py [workflow.yaml] --region SR --no_signal --max_background 10000 --calibrated
```
#### 3.6 Train Weak Supervision Classifier and evaluate the results
#### Full Training [Optional, on slurm]
```aiignore
sh [farm]/run-train_cls.sh
```
#### Local mini-run
```bash
sh [farm]/train_cls-eval.sh
```
The command will look like:
```aiignore
python3 03train_cls.py /global/u1/t/tihsu/CMSOpenData-Upsilon-AnomalyDetection/config/control-boostrap-0.yaml --knumber 2  --drop pc-log_pt-0 pc-log_pt-1 pc-log_energy-0 pc-log_energy-1 pt-balance-pc deltaR-pc pc-phi-0 pc-phi-1 --ignore pfn --test_no_signal --n_gensample 2000 --seed 108614 --calibrated
```
#### 3.7 Final evaluation and plotting
```bash
sh Farm-pretrain/summary-eval.sh
```
This is typically run the final evaluation and plotting for the signal region (SR). The command will look like:
```aiignore
python3 04bump_hunting-eval.py /global/u1/t/tihsu/CMSOpenData-Upsilon-AnomalyDetection/config/control-boostrap-0.yaml  --calibrated
python3 04bump_hunting-eval.py /global/u1/t/tihsu/CMSOpenData-Upsilon-AnomalyDetection/config/control-boostrap-0.yaml --test_no_signal  --calibrated
python3 04bump_hunting-eval.py /global/u1/t/tihsu/CMSOpenData-Upsilon-AnomalyDetection/config/control-boostrap-0.yaml --no_signal  --calibrated
python3 04bump_hunting-eval.py /global/u1/t/tihsu/CMSOpenData-Upsilon-AnomalyDetection/config/control-boostrap-0.yaml --test_no_signal --no_signal  --calibrated
```
This will run the final evaluation and plotting scripts, which will generate the final results and plots under
```text
[outputdir]/[tag]_calibrated_fit
[outputdir]/[tag]_calibrated_fit/plots
```


## References
[1] Rikab Gambhir, Radha Mastandrea, Benjamin Nachman, Jesse Thaler, *Isolating Unisolated Upsilons with Anomaly Detection in CMS Open Data*, Phys. Rev. Lett. 135, 021902 (2025). DOI: [10.1103/vvv3-5kkl.135.021902](https://doi.org/10.1103/vvv3-5kkl.135.021902)
