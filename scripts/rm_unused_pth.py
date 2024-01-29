import os
import shutil
from pathlib import Path

root = Path(__file__).parent.parent.joinpath('output')

for exp in os.listdir(root):
    # print(f'exp={exp}')
    for scene in os.listdir(root.joinpath(exp)):
        # print('scene=', scene)
        pc_dir = root.joinpath(exp, scene, 'point_cloud')
        if not pc_dir.exists():
            continue
        names = os.listdir(pc_dir)
        iterations = [int(name.split('_')[1]) for name in names]
        max_iter = max(iterations)
        for i, name in zip(iterations, names):
            if i == max_iter:
                continue
            print('rm:', pc_dir.joinpath(name))
            shutil.rmtree(pc_dir.joinpath(name))
            if root.joinpath(exp, scene, 'deform').exists():
                shutil.rmtree(root.joinpath(exp, scene, 'deform', name))
