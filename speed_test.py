import os
from argparse import ArgumentParser
from os import makedirs

import imageio
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from gaussian_renderer import render
from scene import Scene, DeformModel, NonRigidDeformationModel
from utils.general_utils import safe_state


@torch.no_grad()
def interpolate_time(
    model_path,
    load2gpt_on_the_fly,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    deform,
    fine_deform,
    num_frames=1000,
    only_static=False,
    no_mlp=True,
):
    render_path = os.path.join(model_path, name)
    makedirs(render_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    results = None
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.synchronize()
    start_time.record()
    for t in tqdm(range(0, num_frames, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (num_frames - 1)]).cuda()
        xyz = gaussians.get_xyz
        time_input = fid.view(-1)  # fid.unsqueeze(0).expand(xyz.shape[0], -1)
        if only_static:
            d_xyz, d_rotation, d_scaling = 0, 0, 0
            fd = None
        else:
            d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, use_mlp=not no_mlp)[0]
            fd = None if fine_deform is None else fine_deform.step(xyz.detach(), d_xyz, d_rotation, time_input)
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, f_deform=fd)["render"]
        # renderings.append(results["render"])
    end_time.record()
    end_time.synchronize()
    t = start_time.elapsed_time(end_time)
    fps = num_frames / (t / 1000.)
    print(f"Rendering {num_frames} images of view {idx} in {t:.2f} ms, fps={fps:.2f}")
    # renderings = np.stack([to8b(img.cpu().numpy()) for img in renderings], 0).transpose(0, 2, 3, 1)
    # imageio.mimwrite(os.path.join(render_path, f'video_{iteration}.mp4'), renderings, fps=60, quality=8)  # noqa
    with open(os.path.join(model_path, name, "results.txt"), 'w') as f:
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Time(ms): {t:.2f}\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"View: {idx}\n")
        f.write(f"only_static: {only_static}")
        f.write(f"use_mlp: {not no_mlp}")


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, only_static, no_mlp):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        deform = DeformModel(
            num_superpoints=dataset.num_superpoints,
            num_knn=dataset.num_knn,
            sp_net_large=dataset.sp_net_large
        )
        deform.load_weights(dataset.model_path)
        if os.path.exists(os.path.join(dataset.model_path, 'fine_deform')):
            fine_deform = NonRigidDeformationModel(small=not model.fine_large, is_blender=dataset.is_blender)
            fine_deform.load_weights(dataset.model_path)
            print('Loaded fine_deform model')
        else:
            fine_deform = None

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        interpolate_time(dataset.model_path, dataset.load2gpu_on_the_fly, "speed", scene.loaded_iter,
            scene.getTestCameras(), gaussians, pipeline, background, deform, fine_deform,
            only_static=only_static, no_mlp=no_mlp
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--no-mlp', action="store_true")
    parser.add_argument('--only-static', action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.only_static, args.no_mlp)
