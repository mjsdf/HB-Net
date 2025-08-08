#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S") # 获取当前日期和时间，格式为 YYYYMMDD_HHMMSS
job='pascal_unimatch_r101_92' # 定义作业名称

# modify these augments if you want to try other datasets, splits or methods
dataset='pascal'
method='unimatch'
exp='r101'
split='92'

config=configs/${dataset}.yaml # 配置文件路径,${dataset} 是一个变量，可能会在后续的代码中根据具体的数据集名称进行替换
labeled_id_path=splits/$dataset/$split/labeled.txt # 有标签数据的 ID 列表路径,$dataset 和 $split 是变量，会在实际使用时被具体的值替换
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt # 无标签数据的 ID 列表路径
save_path=exp/$dataset/$method/$exp/$split # 保存训练输出的路径

mkdir -p $save_path # 如果不存在，则创建保存路径的目录

srun --mpi=pmi2 -p $3 -n $1 --gres=gpu:$1 --ntasks-per-node=$1 --job-name=$job \
    --open-mode=append -o $save_path/$now.log --quotatype=reserved \
    python -u $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2


# 这个脚本是为了在 Slurm 管理的集群上启动一个分布式训练作业
# 通过这个脚本，用户可以在 Slurm 集群上方便地启动和运行分布式训练作业，而无需手动设置所有参数和环境