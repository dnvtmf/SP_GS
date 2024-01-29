import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel, DeformModel, NonRigidDeformationModel
from scene.d_3d_gs import DeformModel as D3DGS_DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.system_utils import searchForMaxIteration

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    testing_iterations,
    saving_iterations,
    teacher_gs,
    teacher_deform,
    teacher_ply_path,
):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians, ply_path=teacher_ply_path)
    gaussians.training_setup(opt)

    train_cameras = scene.getTrainCameras()
    train_times = torch.tensor([cam.fid for cam in train_cameras]).cuda()
    train_times = torch.unique(train_times)
    print(f'There are {len(train_times)} frames')

    deform = DeformModel(num_points=len(gaussians.get_xyz), train_times=train_times,
        num_superpoints=dataset.num_superpoints, num_knn=dataset.num_knn, sp_net_large=dataset.sp_net_large)
    deform.train_setting(opt)
    print(deform.sp_deform)
    print(deform.sp_model)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    for iteration in range(1, opt.iterations + 1):
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
        #                 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        with torch.no_grad():
            xyz = teacher_gs.get_xyz
            d_xyz, d_rotation, d_scaling = teacher_deform.step(xyz.detach(), fid.view(-1))
            teacher_xyz = teacher_gs.get_xyz + d_xyz
            teacher_rot = F.normalize(teacher_gs.get_rotation + d_rotation, dim=-1)
            teacher_scale = teacher_gs.get_scaling + d_scaling

        (d_xyz, d_rotation, d_scaling), loss_aux = deform.step(gaussians.get_xyz.detach(), fid.view(-1), use_mlp=True)

        # Render
        render_pkg_re = render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
        )
        image = render_pkg_re["render"]
        # viewspace_point_tensor = render_pkg_re["viewspace_points"]
        # visibility_filter = render_pkg_re["visibility_filter"]
        # radii = render_pkg_re["radii"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if isinstance(d_xyz, torch.Tensor):
            losses = deform.sp_model.loss(gaussians.get_xyz, render_pkg_re['points'],
                d_rotation, d_xyz, loss_aux, t=fid)
            loss = loss + losses['re_xyz'] * 1e-3
            loss = loss + losses['re_rot']
            loss = loss + losses['re_off']
        # guide loss
        if iteration < opt.warm_up:
            xyz = gaussians.get_xyz + d_xyz
            rotation = F.normalize(gaussians.get_rotation + d_rotation, dim=-1)
            # loss_lambda = min(iteration / opt.iterations, 0.5)
            loss = loss + (F.mse_loss(xyz, teacher_xyz) + F.mse_loss(rotation, teacher_rot)) * 1e-3
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
            #     radii[visibility_filter])

            # Log and save
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background), deform,
                dataset.load2gpu_on_the_fly)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(args.model_path, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
    renderArgs, deform, load2gpu_on_the_fly
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    # time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    # d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), fid.view(-1), use_mlp=True)[0]
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling)[
                            "render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                            image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


def load_teacher(teacher_path):
    teacher_path = Path(teacher_path)
    if teacher_path.is_dir():
        iteration = searchForMaxIteration(teacher_path.joinpath('point_cloud'))
        teacher_ply_path = teacher_path.joinpath('point_cloud', f'iteration_{iteration}', 'point_cloud.ply')
        teacher_pth_path = teacher_path.joinpath('deform', f'iteration_{iteration}', 'deform.pth')
    elif teacher_path.suffix == '.ply':
        teacher_ply_path = teacher_path
        parts = teacher_ply_path.parts
        teacher_pth_path = Path(*parts[:-3], 'deform', parts[-2], 'deform.pth')
    elif teacher_path.suffix == '.pth':
        teacher_pth_path = teacher_path
        parts = teacher_pth_path.parts
        teacher_ply_path = Path(*parts[:-3], 'point_cloud', parts[-2], 'point_cloud.ply')
    else:
        raise ValueError(f"{teacher_path} can not found a D-3D-GS model")
    assert teacher_ply_path.is_file() and teacher_pth_path.is_file()

    gs = GaussianModel(3)
    gs.load_ply(teacher_ply_path)
    pth = torch.load(teacher_pth_path, map_location='cpu')
    is_blender = 'timenet.0.weight' in pth
    deform = D3DGS_DeformModel(is_blender, False)
    deform.deform.load_state_dict(pth)
    deform.deform.cuda()
    print(f'Successfully loaded a D-3D-GS model from {teacher_path}')
    return gs, deform, teacher_ply_path


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--d-3d-gs', type=str)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print(f"Distill D-3D-GS {args.d_3d_gs} to {args.model_path}")

    # Initialize system state (RNG)
    safe_state(args.quiet)
    teacher_gs, teacher_defrom, teacher_ply_path = load_teacher(args.d_3d_gs)
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
        teacher_gs, teacher_defrom, teacher_ply_path)

    # All done
    print("\nTraining complete.")
