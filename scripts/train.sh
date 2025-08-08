#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['pascal', 'cityscapes', 'coco','liefeng']
# method: ['unimatch', 'fixmatch', 'supervised']
# exp: just for specifying the 'save_path'
# split: ['92', '1_16', 'u2pl_1_16', ...,'5labeled']. Please check directory './splits/$dataset' for concrete splits
dataset='liefeng30'
method='unimatch'
#exp='deeplabv3plus_xception'
#exp='pspnet_xception'
#exp='DNA-Net'
exp='unet'
split='160labeled'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log

# Assuming the t-SNE visualization script is named tsne_visualize.py
#python t-SNE.py --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path --save-path $save_path --port $2