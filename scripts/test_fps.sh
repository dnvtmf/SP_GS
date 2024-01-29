#!/usr/bin/env bash

scenes=(bouncingballs  hellwarrior  hook  jumpingjacks lego  mutant  standup  trex)
args=()
for scene in ${scenes[@]}
do
    python speed_test.py -m output/D-NeRF-L/${scene} ${args[@]} --no-mlp
#    python speed_test.py -m output/D-NeRF-fine/${scene} ${args[@]} --no-mlp
#    python speed_test.py -m output/D-NeRF-static/${scene} ${args[@]}
done

scenes=(as basin bell cup plate press sieve)
for scene in ${scenes[@]}
do
  python speed_test.py -m output/NeRF_DS/${scene} ${args[@]} --no-mlp
#  python speed_test.py -m output/NeRF_DS_fine/${scene} ${args[@]} --no-mlp
#  python speed_test.py -m output/NeRF_DS_static/${scene} ${args[@]}
done

scenes=(vrig-3dprinter vrig-broom vrig-chicken vrig-peel-banana)
for scene in ${scenes[@]}
do
  python speed_test.py -m output/HyperNeRF/${scene} ${args[@]} --no-mlp
#  python speed_test.py -m output/HyperNeRF-fine/${scene} ${args[@]} --no-mlp
#  python speed_test.py -m output/HyperNeRF-static/${scene} ${args[@]}
done
