import argparse
import os
import yaml
from copy import deepcopy


def absolutize_path(path_value, base_dir):
    if path_value is None:
        return None
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(base_dir, path_value))


def prepare_script(args):
    os.makedirs(args.farm, exist_ok=True)
    work_dir = os.getcwd()
    with open(args.train_config) as f:
        config = yaml.safe_load(f)
    config_dir = os.path.dirname(args.train_config)
    print(config_dir)
    command = []
    init = 0.0
    step = 1.0 / args.fold

    abs_evenet_dir = os.path.abspath(args.evenet_dir)

    original_wandb_run_name = config['logger']['wandb']['run_name']
    for ifold in range(args.fold):
        file_path = os.path.join(work_dir, args.farm, f"fold_{ifold}.yaml")
        fold_config = deepcopy(config)
        fold_config['network']['default'] = os.path.join(work_dir, config_dir, fold_config['network']['default'])
        fold_config['event_info']['default'] = os.path.join(work_dir, config_dir, fold_config['event_info']['default'])
        fold_config['resonance']['default'] = os.path.join(work_dir, config_dir, fold_config['resonance']['default'])
        fold_config['options']['default'] = os.path.join(work_dir, config_dir, fold_config['options']['default'])
        fold_config['logger']['wandb']['run_name'] = f"{original_wandb_run_name}-fold{ifold}"

        fold_config["platform"]["data_parquet_dir"] = absolutize_path(
            fold_config["platform"]["data_parquet_dir"], config_dir
        )
        if "data_parquet_val_dir" in fold_config["platform"]:
            fold_config["platform"]["data_parquet_val_dir"] = absolutize_path(
                fold_config["platform"]["data_parquet_val_dir"], config_dir
            )

        fold_config["options"]["Dataset"]["normalization_file"] = absolutize_path(
            fold_config["options"]["Dataset"]["normalization_file"], config_dir
        )
        fold_config["options"]["Training"]["model_checkpoint_save_path"] = absolutize_path(
            fold_config["options"]["Training"]["model_checkpoint_save_path"], config_dir
        )
        if fold_config["options"]["Training"].get("model_checkpoint_load_path") is not None:
            fold_config["options"]["Training"]["model_checkpoint_load_path"] = absolutize_path(
                fold_config["options"]["Training"]["model_checkpoint_load_path"], config_dir
            )
        if fold_config["options"]["Training"].get("pretrain_model_load_path") is not None:
            fold_config["options"]["Training"]["pretrain_model_load_path"] = absolutize_path(
                fold_config["options"]["Training"]["pretrain_model_load_path"], config_dir
            )

        fold_config["options"]["Training"]["model_checkpoint_save_path"] = os.path.join(fold_config["options"]["Training"]["model_checkpoint_save_path"], f"fold_{ifold}")
        fold_config["options"]["Dataset"]["val_split"] = [init, init + step]
        init += step
        with open(file_path, 'w') as f:
            yaml.dump(fold_config, f)

        if args.local:
            command.append(f"cd {abs_evenet_dir}; python3 scripts/train.py {file_path} --load_all --ray_dir {args.ray_dir}")
        else:
            command.append(f"cd {abs_evenet_dir}; shifter --image={args.image} python3 scripts/train.py {file_path} --load_all --ray_dir {args.ray_dir}")

    # Write the command to a shell script
    script_path = os.path.join(args.farm, "train.sh")
    with open(script_path, 'w') as f:
        # f.write("#!/bin/bash\n")
        for cmd in command:
            f.write(f"{cmd}\n")



def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type = str)
    parser.add_argument("--farm", type = str)
    parser.add_argument("--fold", type = int, default = 5)
    parser.add_argument("--ray_dir", type=str, default = "~/ray_results")
    parser.add_argument("--local", action='store_true', help="Run locally without shifter")
    parser.add_argument("--evenet-dir", type=str, default = "EveNet-Full")
    parser.add_argument("--image", type=str)
    # Parse command-line arguments
    args = parser.parse_args()
    # Explore the provided HDF5 file
    prepare_script(args)

if __name__ == "__main__":
    main()
