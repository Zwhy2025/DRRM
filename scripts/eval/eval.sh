#!/bin/bash
task=block_hammer_beat
policy=vodp

# ckpt_name=vodp-ur12e-small-100-checkpoint_40000
# ckpt_dir="/root/workspace/$ckpt_name"

ckpt_name=checkpoint-40000
ckpt_dir="/root/workspace/DRRM/checkpoints/vodp_block_hammer_beat_100_23d_1f_/$ckpt_name"

num_process=1
round_num=1

for ((i=1; i<=round_num; i+=1)); do
    exp_num=exp$i
    save_dir="eval_result/$ckpt_name/$exp_num"

    python simulation/robotwin/script/eval_policy_$policy.py \
        --checkpoint-dir $ckpt_dir \
        --save-dir $save_dir \
        --task-name $task \
        --num-process $num_process\
        --seed 0
    
        while IFS= read -r -d '' file; do
            filename=$(basename "$file")
            souc=$save_dir/$filename
            dest="${save_dir%/$exp_num}/${exp_num}_${filename}"
            echo "$souc -> $dest"
        cp $souc $dest
    done < <(find "$save_dir" -maxdepth 1 -type f -print0)
done