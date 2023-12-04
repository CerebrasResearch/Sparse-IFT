#!/bin/bash

seed=${seed:-0}
lr=${learning_rate:-"0.1"}
model=${model:-"resnet_18"}
data=${data:-"cifar100"}
epochs=${epochs:-"200"}
batch_size=${batch_size:-"128"}
log_interval=${log_interval:-"100"}
weight_decay=${weight_decay:-"5e-4"}
output_dir=${output_dir:-"/cb/home/vithu/extra-storage/want/resnet18"}
num_steps=${num_steps:-1}
block_name=${block_name:-"Block"}
optimizer=${optimizer:-"sgd"}
momentum=${momentum:-"0.9"}

# Sparsity
############
maskopt=${maskopt:-"None"}
maskopt_begin_iteration=${maskopt_begin_iteration:-0}
maskopt_frequency=${maskopt_frequency:-100}
maskopt_sparsity=${maskopt_sparsity:-0.5}
maskopt_mask_init_method=${maskopt_mask_init_method:-"random"}
maskopt_mask_distribution=${maskopt_mask_distribution:-"uniform"}
flop_budget=${flop_budget:-"1115226800"} # budget for ResNet18 v1.5
fused_bn=${fused_bn:-"false"}
enable_head=${enable_head:-"false"}
enable_stem=${enable_stem:-"false"}

# sift params
sift_scaling=${sift_scaling:-"false"}
sift_family=${sift_family:-"sparse_wide"}
sp_use_internal_relu6=${sp_use_internal_relu6:-"false"}
sp_use_layer_specialization=${sp_use_layer_specialization:-"false"}
sparse_u_block=${sparse_u_block:-"false"}
uv_layer=${uv_layer:-"None"}
sparse_width_scaling=${sparse_width_scaling:-0.0}

# Build the command
############
CMD="/cb/home/vithu/anaconda3/envs/want/bin/python -u main.py "
CMD+=" --seed=$seed"
CMD+=" --lr=$lr"
CMD+=" --model=$model"
CMD+=" --data=$data"
CMD+=" --num_steps=$num_steps"
CMD+=" --epochs=$epochs"
CMD+=" --batch_size=$batch_size"
CMD+=" --log_interval=$log_interval"
CMD+=" --output_dir=$output_dir"
CMD+=" --weight_decay=$weight_decay"
CMD+=" --input_size 3 32 32"
CMD+=" --block_name=$block_name"
CMD+=" --flop_budget=$flop_budget"
# note that filtering decays from optim, nesterov and zero-init are switched on
# by default at this stage -- you can avoid passing in arguments for now
CMD+=" --lr_scheduler multistep --lr_gamma 0.2 --lr_milestones 60 120 160"
CMD+=" --optimizer=$optimizer"
CMD+=" --momentum=$momentum"

# Sparsity
############
if [ "$maskopt" != "None" ] ; then
    CMD+=" --maskopt=$maskopt"
    CMD+=" --maskopt_begin_iteration=$maskopt_begin_iteration"
    CMD+=" --maskopt_drop_fraction=0.3"
    CMD+=" --maskopt_frequency=100"
    CMD+=" --maskopt_sparsity=$maskopt_sparsity"
    CMD+=" --maskopt_mask_init_method=$maskopt_mask_init_method"
    CMD+=" --maskopt_mask_distribution=$maskopt_mask_distribution"
    CMD+=" --flop_budget=$flop_budget"

    if [ "$fused_bn" == "true" ] ; then
        CMD+=" --maskopt_fused_bn_updates"
    fi

    if [ "$enable_head" == "true" ] ; then
        CMD+=" --maskopt_sparsity_head"
    fi

    if [ "$enable_stem" == "true" ] ; then
        CMD+=" --maskopt_sparsity_stem"
    fi
fi

if [ "$block_name" == "BlockISOFlopWideFactorized" ] ; then
    CMD+=" --sparse_width_scaling=$sparse_width_scaling"
fi

# sift scaling
if [ "$sift_scaling" == "True" ] ; then
    CMD+=" --sift_scaling"
    CMD+=" --sift_family=$sift_family"

    if [ "$sp_use_internal_relu6" == "True" ] ; then
        CMD+=" --sp_use_internal_relu6"
    fi

    if [ "$sp_use_layer_specialization" == "True" ] ; then
        CMD+=" --sp_use_layer_specialization"
    fi

    if [ "$sparse_u_block" ==  "True" ]; then
        CMD+=" --sparse_u_block"
    fi

    if [ "$uv_layer" != "None" ]; then
        CMD+=" --uv_layer=$uv_layer"
    fi
fi

echo $CMD

$CMD
