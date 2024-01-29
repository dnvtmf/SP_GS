import json
import os
from pathlib import Path
import rich
from rich.console import Console
from rich.table import Table

console = Console()
root = Path(__file__).parent.parent
datsets = list(root.joinpath('output').iterdir())
for db in sorted(datsets):
    if not os.path.isdir(db):
        continue
    scenes = [scene for scene in sorted(os.listdir(db)) if db.joinpath(scene, 'results.json').exists()]
    if len(scenes) == 0:
        continue
    table = Table(title=f"Results for {db.name}")
    table.add_column()
    results = []
    for scene in scenes:
        if db.joinpath(scene, 'results.json').exists():
            with open(db.joinpath(scene, 'results.json'), 'r') as f:
                res = json.load(f)
                last_iter = max(int(k.split('_')[1]) for k in res.keys())
                results.append(res[f"ours_{last_iter}"])
        else:
            results.append({})
        if db.joinpath(scene, 'speed/results.txt').exists():
            with open(db.joinpath(scene, 'speed/results.txt'), 'r') as f:
                line = f.readline()
                # print(line)
                FPS = float(line.strip().split(':')[1])
            results[-1]['FPS'] = FPS
        table.add_column(scene)
    table.add_column('average')
    for metric in ['PSNR', 'SSIM', 'MS-SSIM', "LPIPS (VGG)", 'LPIPS (Alex)', 'FPS']:
        row = [metric]
        total = 0
        num = 0
        for res_s in results:
            # print(res_s)
            if metric in res_s:
                row.append('{:.4f}'.format(res_s[metric]))
                total += res_s[metric]
                num += 1
            else:
                row.append('-')
        if total == 0:
            row.append('N/A')
        else:
            row.append('{:.4f}'.format(total / num))
        table.add_row(*row)
    console.print(table)
