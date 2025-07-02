#!/bin/bash

env=$1
l2=$2
cost_loss=$3
delay=$4
segment_length=$5
cost_loss_modifier=$6
cost_loss_bias=$7
coef=$8
policy_noise=${9}
noise_clip=${10}
cost_cap=${11}
lr=${12}
error_cap=${13}

buffer_id=${14}
cost_version=${15}
intervention_threshold=${16}

seed=${17}
custom_id=${18}


module load cuda cudnn

source "/home/xiruzhu/tf_2_5/bin/activate"
python train_model.py --env_id=$env --vf_lr=$lr  --seed=$seed --custom_id=$custom_id --error_function_cap=$error_cap --l2=$l2 --cost_loss_modifier=$cost_loss_modifier --cost_loss_bias=$cost_loss_bias --intervention_loss=$cost_loss --segment_length=$segment_length --intervention_delay=$delay --policy_noise=$policy_noise --noise_clip=$noise_clip --cost_only=1 --cost_cap=$cost_cap --coefficient=$coef --buffer_id=$buffer_id --cost_version=$cost_version --intervention_threshold=$intervention_threshold

