#!/usr/bin/env bash
scenes=(vrig-3dprinter vrig-broom vrig-chicken vrig-peel-banana)
gpus=(0 1 2 3)

args=(--warm_up 0 --iterations 20000 --densify_from_iter 30000 --densify_until_iter -1 \
 --deform_lr_max_steps 20000 --position_lr_max_steps 20000)
test_args=()
num_scenes=${#scenes[@]}
num_gpus=${#gpus[@]}
out_dir=HyperNeRF-fine
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
      "python3 train_fine.py -s data/HyperNeRF/vrig/${scenes[i]} -m output/${out_dir}/${scenes[i]} \
        --load output/HyperNeRF/${scenes[i]} --eval ${args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 render.py -m output/${out_dir}/${scenes[i]} --skip_train ${test_args[*]} ^M"
    screen -S gpu${gpu_id} -p 0 -X stuff \
      "python3 metrics.py -m output/${out_dir}/${scenes[i]} ^M"
done
