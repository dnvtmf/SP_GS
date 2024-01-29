#!/usr/bin/env bash
scenes=(bouncingballs  hellwarrior  hook  jumpingjacks lego  mutant  standup  trex)
gpus=(0 1 2 3)
#scenes=(hellwarrior)
#gpus=(1)
#args=(--warm_up 0 --iterations 20000 --densify_from_iter 40000 --densify_until_iter -1 \
#  --deform_lr_max_steps 20000 --position_lr_max_steps 20000 --sp_net_large)
args=()
test_args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
out_dir="D-NeRF-D3DGS"
echo "There are ${num_gpus} gpus and ${num_scenes} scenes"

for (( i = 0;  i < ${num_gpus}; ++i ))
do
    gpu_id="gpu${gpus[$i]}"
    if ! screen -ls ${gpu_id}
    then
        echo "create ${gpu_id}"
        screen -dmS ${gpu_id}
    fi
    screen -S ${gpu_id} -p 0 -X stuff "^M"
    screen -S ${gpu_id} -p 0 -X stuff "export CUDA_VISIBLE_DEVICES=${gpus[$i]}^M"
    screen -S ${gpu_id} -p 0 -X stuff "cd ~/Projects/NeRF/SP_GS^M"
done
screen -ls%

for (( i=0; i < num_scenes; ++i ))
do
    gpu_id=${gpus[$(( i % num_gpus ))]}
    echo "use gpu${gpu_id} on scene: ${scenes[i]} "
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 train_fine.py -s data/D-NeRF/${scenes[i]} -m output/${out_dir}/${scenes[i]} \
        --eval --fine_large --is_blender ${args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 render.py -m output/${out_dir}/${scenes[i]} --skip_train ${test_args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 metrics.py -m output/${out_dir}/${scenes[i]} ^M"
done
