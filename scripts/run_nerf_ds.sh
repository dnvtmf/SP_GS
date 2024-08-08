#!/usr/bin/env bash
scenes=(as basin bell cup plate press sieve)
gpus=(0 1 2 3 5 6 8 9)
args=(--test_iterations 5000 10000 20000 30000 40000)
test_args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
out_dir=NeRF-DS
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
    screen -S ${gpu_id} -p 0 -X stuff "conda activate sp_gs_env^M"
done
screen -ls%

for (( i=0; i < num_scenes; ++i ))
do
    gpu_id=${gpus[$(( i % num_gpus ))]}
    echo "use gpu${gpu_id} on scene: ${scenes[i]} "
    screen -S gpu${gpu_id} -p 0 -X stuff "^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 train.py -s data/NeRF-DS/${scenes[i]} -m output/${out_dir}/${scenes[i]} --eval ${args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 render.py -m output/${out_dir}/${scenes[i]} -s data/NeRF-DS/${scenes[i]} \
        --skip_train ${test_args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 metrics.py -m output/${out_dir}/${scenes[i]} ^M"
done
