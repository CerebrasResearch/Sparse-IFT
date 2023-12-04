#!/bin/bash

# data params
data_path=${data_path:-"/cb/datasets/cv/imagenet/imagenet1k_ilsvrc2012/"}
interpolation=${interpolation:-"bilinear"}
val_resize_size=${val_resize_size:-"256"}
val_crop_size=${val_crop_size:-"224"}
train_crop_size=${train_crop_size:-"224"}
batch_size=${batch_size:-"32"}
workers=${workers:-"8"}
num_gpus=${num_gpus:-"8"}

# model params
model=${model:-"resnet_18"}
block_name=${block_name:-"Block"}

# optimizer params
optimizer=${optimizer:-"sgd"}
lr=${lr:-"0.1"}
momentum=${momentum:-"0.9"}
weight_decay=${weight_decay:-"1e-4"}
label_smoothing=${label_smoothing:-"0.0"}
gradient_accumulation_steps=${gradient_accumulation_steps:-1}
epochs=${epochs:-90}

# learning rate params
lr_scheduler=${lr_scheduler:-"cosineannealing"}
lr_min=${lr_min:-"0.0"}

# misc params
print_freq=${print_freq:-"500"}
output_dir=${output_dir:-"/tmp/dense"}
seed=${seed:-0}
amp=${amp:-"false"}
sync_bn=${sync_bn:-"false"}
clip_grad_norm=${clip_grad_norm:-"None"}
resume_ckpt=${resume_ckpt:-""}

# sparsity params
maskopt=${maskopt:-"None"}
maskopt_begin_iteration=${maskopt_begin_iteration:-0}
maskopt_frequency=${maskopt_frequency:-100}
maskopt_sparsity=${maskopt_sparsity:-0.5}
maskopt_mask_init_method=${maskopt_mask_init_method:-"random"}
maskopt_mask_distribution=${maskopt_mask_distribution:-"uniform"}
flop_budget=${flop_budget:-"3647128359"} # budget for ResNet18 v1.5
enable_head=${enable_head:-"false"}
enable_stem=${enable_stem:-"false"}

# sift params
sift_scaling=${sift_scaling:-"false"}
sift_family=${sift_family:-"sparse_wide"}


# Build the command
############
CMD="/cb/home/vithu/anaconda3/envs/want/bin/torchrun --standalone --nproc_per_node=$num_gpus main.py"
CMD+=" --model=$model"
CMD+=" --block_name=$block_name"
CMD+=" --input_size 3 224 224"
CMD+=" --data_path=$data_path"
CMD+=" --interpolation=$interpolation"
CMD+=" --val_resize_size=$val_resize_size"
CMD+=" --val_crop_size=$val_crop_size"
CMD+=" --train_crop_size=$train_crop_size"
CMD+=" --batch_size=$batch_size"
CMD+=" --workers=$workers"
CMD+=" --lr_scheduler=$lr_scheduler"
CMD+=" --lr_min=$lr_min"
CMD+=" --print_freq=$print_freq"
CMD+=" --output_dir=$output_dir"
CMD+=" --seed=$seed"
CMD+=" --optimizer=$optimizer"
CMD+=" --lr=$lr"
CMD+=" --momentum=$momentum"
CMD+=" --weight_decay=$weight_decay"
CMD+=" --label_smoothing=$label_smoothing"
CMD+=" --gradient_accumulation_steps=$gradient_accumulation_steps"
CMD+=" --epochs=$epochs"
CMD+=" --resume_ckpt=$resume_ckpt"

if [ "$amp" == "True" ] ; then
    CMD+=" --amp"
fi

if [ "$sync_bn" == "True" ] ; then
    CMD+=" --sync_bn"
fi

if [ "$clip_grad_norm" != "None" ] ; then
    CMD+=" --clip_grad_norm=$clip_grad_norm"
fi

# sparsity
if [ "$maskopt" != "None" ] ; then
    CMD+=" --maskopt=$maskopt"
    CMD+=" --maskopt_begin_iteration=$maskopt_begin_iteration"
    CMD+=" --maskopt_drop_fraction=0.3"
    CMD+=" --maskopt_frequency=100"
    CMD+=" --maskopt_sparsity=$maskopt_sparsity"
    CMD+=" --maskopt_mask_init_method=$maskopt_mask_init_method"
    CMD+=" --maskopt_mask_distribution=$maskopt_mask_distribution"
    CMD+=" --flop_budget=$flop_budget"

    if [ "$enable_head" == "true" ] ; then
        CMD+=" --maskopt_sparsity_head"
    fi

    if [ "$enable_stem" == "true" ] ; then
        CMD+=" --maskopt_sparsity_stem"
    fi
fi

# sift scaling
if [ "$sift_scaling" == "True" ] ; then
    CMD+=" --sift_scaling"
    CMD+=" --sift_family=$sift_family"
fi

echo $CMD

$CMD
