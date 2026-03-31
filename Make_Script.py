import os, sys
sys.path.append(os.path.abspath("../.."))

import yaml
import os
import vector
vector.register_awkward()
import argparse
from helpers.utils import clean_and_append
import random

from copy import deepcopy


def prepare_script(args):

    with open(args.config_workflow, 'r') as f:
        control_template = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(args.config_workflow))
    os.makedirs(args.farm, exist_ok=True)

    cwd = os.getcwd()


    f_prepare = open(f"{args.farm}/prepare.sh", "w")
    f_prepare.write("#!/bin/bash\n")

    f_train = open(f"{args.farm}/train.sh", "w")
    f_train.write("#!/bin/bash\n")
    f_train.write("set -euo pipefail\n")

    f_predict = open(f"{args.farm}/predict.sh", "w")

    f_train_cls_prepare = open(f"{args.farm}/train_cls_prepare.sh", "w")
    f_train_cls_prepare.write("#!/bin/bash\n")
    f_train_cls_prepare.write(f'cd {cwd}\n')

    f_train_cls = open(f"{args.farm}/train_cls.sh", "w")
    f_train_cls.write(f'cd {cwd}\n')

    f_train_cls_eval = open(f"{args.farm}/train_cls-eval.sh", "w")
    f_train_cls_eval.write(f'cd {cwd}\n')

    f_summary = open(f"{args.farm}/summary.sh", "w")
    f_summary.write("#!/bin/bash\n")

    f_summary_eval = open(f"{args.farm}/summary-eval.sh", "w")
    f_summary_eval.write("#!/bin/bash\n")

    train_list = []
    for iboostrap in range(args.boostrap):
        f_prepare.write(f"(\n")
        control = deepcopy(control_template)

        control['output']['plotdir'] = os.path.join(control['output']['plotdir'], f"boostrap_{iboostrap}")
        control['output']['storedir'] = os.path.join(control['output']['storedir'], f"boostrap_{iboostrap}")
        control['wandb'] = f"{control['wandb']}_boostrap_{iboostrap}"

        config_workflow_file = os.path.join(base_dir, f"control-boostrap-{iboostrap}.yaml")
        with open(config_workflow_file, 'w') as f_boost:
            yaml.dump(control, f_boost, default_flow_style=False)

        f_prepare.write(f'cd {cwd}\n ')
        f_prepare.write(f'python3 00prepare_CMSOpenData.py {config_workflow_file}\n')
        storedir = control["output"]["storedir"]

        os.chdir(base_dir)
        with open(control["train"]["config"], 'r') as f_config:
            train_config = yaml.safe_load(f_config)
        train_config["platform"]["data_parquet_dir"] = f"{storedir}/SB"
        train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{storedir}/SB/model_checkpoint"
        train_config["options"]["Dataset"]["normalization_file"] = f"{storedir}/SB/normalization.pt"
        train_config['logger']['wandb']['run_name'] = f"{control['wandb']}_boostrap_{iboostrap}_OS"
        train_config['platform']['number_of_workers'] = args.gpu

        with open(control["train"]["config"].replace(".yaml", f"-boostrap-{iboostrap}.yaml"), 'w') as f_config:
            yaml.dump(train_config, f_config, default_flow_style=False)

        if args.no_signal:
            no_signal_storedir = clean_and_append(storedir, "_no_signal")
            train_config["platform"]["data_parquet_dir"] = f"{no_signal_storedir}/SB"
            train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{no_signal_storedir}/SB/model_checkpoint"
            train_config["options"]["Dataset"]["normalization_file"] = f"{no_signal_storedir}/SB/normalization.pt"
            train_config['logger']['wandb']['run_name'] = f"{train_config['logger']['wandb']['run_name']}".replace("_OS", "_SS")
            with open(control["train"]["config"].replace(".yaml", f"-boostrap-{iboostrap}-nosig.yaml"), 'w') as f_config:
                yaml.dump(train_config, f_config, default_flow_style=False)


        f_prepare.write(f'python3 03kfold_train.py {os.path.abspath(control["train"]["config"].replace(".yaml", f"-boostrap-{iboostrap}.yaml"))} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm}-gen/boostrap-{iboostrap} --image {control["image"]} \n ')
        if args.no_signal:
            f_prepare.write(f'python3 03kfold_train.py {os.path.abspath(control["train"]["config"].replace(".yaml", f"-boostrap-{iboostrap}-nosig.yaml"))} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm}-gen-nosignal/boostrap-{iboostrap}  --image {control["image"]} \n')

        os.chdir(cwd)
        train_list.append(os.path.abspath(f"{args.farm}-gen/boostrap-{iboostrap}/train.sh"))
        if args.no_signal:
            train_list.append(os.path.abspath(f"{args.farm}-gen-nosignal/boostrap-{iboostrap}/train.sh"))


        # f_predict.write(f'python3 01predict_SR.py {os.path.abspath(config_workflow_file)} --region SB --checkpoint {f"{storedir}/SB/model_checkpoint"} --gen_num_events {args.gen_events} --ngpu {args.gpu} --kfold {args.k}\n')
        for k in range(args.k):
            for ngpu in range(args.gpu):
                f_predict.write(f'python3 01predict_SR_certain_fold.py {os.path.abspath(config_workflow_file)} --region SR --checkpoint {f"{storedir}/SB/model_checkpoint"}  --gen_num_events {int(args.gen_events / (args.k * args.gpu))} --ngpu 1 --rank {ngpu} --fold {k}\n')
                #   f_predict.write(f'python3 01predict_SR.py {os.path.abspath(args.config_workflow)} --region SB --checkpoint {f"{no_signal_storedir}/SB/model_checkpoint"} --gen_num_events {args.gen_events} --ngpu {args.gpu} --kfold {args.k} --no_signal\n')
                if args.no_signal:
                    f_predict.write(f'python3 01predict_SR_certain_fold.py {os.path.abspath(config_workflow_file)} --region SR --checkpoint {f"{no_signal_storedir}/SB/model_checkpoint"} --gen_num_events  {int(args.gen_events / (args.k * args.gpu))} --ngpu 1 --rank {ngpu} --fold {k} --no_signal\n')

        drop_str = f"--drop {' '.join(args.drop)}" if len(args.drop) > 0 else ""
        only_pc_str = "--only_pc" if args.only_pc else ""

        calibrated_command = "--calibrated" if args.calibrated else ""


        random_seed = random.sample(range(1, 400001), args.num_toys)  # 10 unique numbers between 1 and 100
        for seed in random_seed:
            f_train_cls_eval.write(
                f'python3 03train_cls.py {os.path.abspath(config_workflow_file)} --knumber {args.k} {only_pc_str} {drop_str} --ignore pfn --test_no_signal --n_gensample {args.max_background} --seed {seed} {calibrated_command}\n')
            f_train_cls_eval.write(
                f'python3 03train_cls.py {os.path.abspath(config_workflow_file)} --knumber {args.k} {only_pc_str} {drop_str} --ignore pfn --no_signal --test_no_signal --n_gensample {args.max_background} --seed {seed} {calibrated_command}\n')


        f_train_cls_prepare.write(f'python3 02prepare_classification_dataset.py {os.path.abspath(config_workflow_file)} --region SR --max_background {args.max_background * args.num_toys} {calibrated_command}\n')
        f_train_cls_prepare.write(f'python3 02prepare_classification_dataset.py {os.path.abspath(config_workflow_file)} --region SR --no_signal --max_background {args.max_background * args.num_toys} {calibrated_command} \n')
        f_train_cls.write(f'python3 03train_cls.py {os.path.abspath(config_workflow_file)} --knumber {args.k} {only_pc_str} {drop_str} --ignore pfn --test_no_signal\n')
        f_train_cls.write(f'python3 03train_cls.py {os.path.abspath(config_workflow_file)} --knumber {args.k} {only_pc_str} {drop_str} --ignore pfn --no_signal --test_no_signal\n')

        #
        # os.chdir(base_dir)
        # with open(control["train-cls"]["config"], 'r') as f_config:
        #     train_config = yaml.safe_load(f_config)
        #
        # cls_store_dir = clean_and_append(storedir, "_hybrid")
        # train_config["platform"]["data_parquet_dir"] = f"{cls_store_dir}/SR"
        # train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{cls_store_dir}/SR/model_checkpoint"
        # train_config["options"]["Dataset"]["normalization_file"] = f"{cls_store_dir}/SR/normalization.pt"
        # train_config['wandb']['run_name'] = f"{train_config['wandb']['run_name']}_pretrain"
        # with open(control["train-cls"]["config"], 'w') as f_config:
        #     yaml.dump(train_config, f_config, default_flow_style=False)
        #
        #
        #
        # cls_train_config_path = os.path.abspath(control["train-cls"]["config"])
        #
        # train_config["options"]["Training"]["pretrain_model_load_path"] = None
        # train_config['wandb']['run_name'] = f"{train_config['wandb']['run_name']}_scratch"
        # train_config["options"]["Training"]["model_checkpoint_save_path"] = f"{cls_store_dir}/SR/model_checkpoint_scratch"
        #
        #
        # with open(control["train-cls"]["config"].replace(".yaml", "_scratch.yaml"), 'w') as f_config:
        #     yaml.dump(train_config, f_config, default_flow_style=False)
        #
        # cls_train_scratch_path = os.path.abspath(control["train-cls"]["config"].replace(".yaml", "_scratch.yaml"))
        #
        # os.chdir(cwd)
        #
        # farm_dir = os.path.abspath(args.farm)
        # farm_scratch_dir = os.path.abspath(args.farm + "_scratch")
        #
        f_summary.write(f'python3 04bump_hunting.py {os.path.abspath(config_workflow_file)} {"--plot_only" if args.plot_only else ""}  \n')
        f_summary.write(f'python3 04bump_hunting.py {os.path.abspath(config_workflow_file)} --test_no_signal {"--plot_only" if args.plot_only else ""}\n')
        f_summary.write(f'python3 04bump_hunting.py {os.path.abspath(config_workflow_file)} --test_no_signal --no_signal {"--plot_only" if args.plot_only else ""}\n')
        f_summary.write(f'python3 04bump_hunting.py {os.path.abspath(config_workflow_file)} --no_signal {"--plot_only" if args.plot_only else ""}\n')


        # f_summary_eval.write(f'python3 04bump_hunting-eval.py {os.path.abspath(config_workflow_file)} {"--plot_only" if args.plot_only else ""}  \n')
        f_summary_eval.write(f'python3 04bump_hunting-eval.py {os.path.abspath(config_workflow_file)} {"--plot_only" if args.plot_only else ""} {calibrated_command}\n')
        f_summary_eval.write(f'python3 04bump_hunting-eval.py {os.path.abspath(config_workflow_file)} --test_no_signal {"--plot_only" if args.plot_only else ""} {calibrated_command}\n')
        f_summary_eval.write(f'python3 04bump_hunting-eval.py {os.path.abspath(config_workflow_file)} --no_signal {"--plot_only" if args.plot_only else ""} {calibrated_command}\n')
        f_summary_eval.write(f'python3 04bump_hunting-eval.py {os.path.abspath(config_workflow_file)} --test_no_signal --no_signal {"--plot_only" if args.plot_only else ""} {calibrated_command}\n')

        # # f.write(f'python3 03kfold_train.py {cls_train_config_path} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm} {"--local" if args.local else ""} \n')
        # # f.write(f'python3 03kfold_train.py {cls_train_scratch_path} --fold {args.k} --ray_dir {args.ray_dir} --farm {args.farm + "_scratch"} {"--local" if args.local else ""} \n')
        # # f.write(f'cd {control["workdir"]} \n')
        # # f.write(f'sh {farm_dir}/train.sh \n')
        # # f.write(f'sh {farm_scratch_dir}/train.sh \n')

        f_prepare.write(") & \n")

    train_command_file = os.path.abspath(os.path.join(args.farm, "train_commands.txt"))
    submit_script = os.path.abspath(os.path.join(cwd, "script", "submit_multiple_ray.sh"))
    cpu_submit_script = os.path.abspath(os.path.join(cwd, "script", "run_on_ncpus.sh"))
    tasks_per_node = max(1, int(args.total_gpu / args.gpu))

    f_train.write(f'train_command_file="{train_command_file}"\n')
    f_train.write("train_sources=(\n")
    for train_script in train_list:
        f_train.write(f'"{train_script}"\n')
    f_train.write(")\n")
    f_train.write('> "$train_command_file"\n')
    f_train.write('for src in "${train_sources[@]}"; do\n')
    f_train.write('  if [ ! -f "$src" ]; then\n')
    f_train.write('    echo "Missing generated train script: $src" >&2\n')
    f_train.write('    echo "Run [farm]/prepare.sh first to generate the k-fold training scripts." >&2\n')
    f_train.write('    exit 1\n')
    f_train.write('  fi\n')
    f_train.write('  cat "$src" >> "$train_command_file"\n')
    f_train.write("  printf '\\n' >> \"$train_command_file\"\n")
    f_train.write('done\n')
    f_train.write('if [ ! -s "$train_command_file" ]; then\n')
    f_train.write('  echo "No training commands were generated." >&2\n')
    f_train.write('  exit 1\n')
    f_train.write('fi\n')
    f_train.write(
        f'sh "{submit_script}" --gpus-per-task {args.gpu} --ntasks 1 '
        f'--tasks-per-node {tasks_per_node} "$train_command_file"\n'
    )

    f_prepare.close()
    f_train.close()
    f_train_cls_prepare.close()
    f_train_cls.close()
    f_train_cls_eval.close()
    f_predict.close()
    f_summary.close()
    f_summary_eval.close()
    with open(f"{args.farm}/run-predict.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(
            f'sh "{submit_script}" --gpus-per-task 1 --ntasks 1 '
            f'--tasks-per-node {args.total_gpu} {os.path.abspath(args.farm)}/predict.sh\n'
        )


    with open(f"{args.farm}/run-train_cls.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(
            f'sh "{cpu_submit_script}" --cpus-per-task 4 '
            f'{os.path.abspath(args.farm)}/train_cls.sh\n'
        )
    with open(f"{args.farm}/run-summary.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(
            f'sh "{cpu_submit_script}" --cpus-per-task {args.cpu} '
            f'{os.path.abspath(args.farm)}/summary.sh\n'
        )
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("config_workflow", type = str)
    parser.add_argument("--boostrap", type = int, default = 1, help="Bootstrap number")
    parser.add_argument("--farm", type = str, default = "Farm")
    parser.add_argument("--ray_dir", type = str, default = "/pscratch/sd/t/tihsu/tmp")
    parser.add_argument("--gen_events", type = int, default = 102400)
    parser.add_argument("--cpu", type = int, default = 10, help="Number of CPUs per task")
    parser.add_argument("--total-gpu", type=int, default = 1)
    parser.add_argument("--gpu", type = int, default = 1)
    parser.add_argument("--max_background", type = int, default = 40000)
    parser.add_argument("--k", type = int, default = 3)
    parser.add_argument("--local", action='store_true', help="Run locally instead of on the farm")
    parser.add_argument("--only_pc", action='store_true')
    parser.add_argument("--drop", nargs = "+", type = str)
    parser.add_argument("--no_signal", action='store_true', help="Do not use signal processing features")
    parser.add_argument("--test_no_signal", action='store_true', help="Test mode for no signal processing features")
    parser.add_argument("--plot_only", action='store_true', help="Only plot results without training or prediction")
    parser.add_argument("--num_toys", type = int, default = 200, help="Number of toys for classification training")
    parser.add_argument("--calibrated", action='store_true', help="Use calibrated mass for classification training")
    # Parse command-line arguments
    args = parser.parse_args()

    prepare_script(args)

if __name__ == "__main__":
    main()
