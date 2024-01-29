import os

import torch
import torch.nn as nn
from lietorch import SO3

from utils.general_utils import get_expon_lr_func
from utils.system_utils import searchForMaxIteration
from utils.time_utils import DeformNetwork


class NonRigidDeformationModel:
    def __init__(self, small=True, is_blender=False):
        if small:
            self.deform = DeformNetwork(W=64, D=3, skips=(), is_blender=False).cuda()
        else:
            self.deform = DeformNetwork(is_blender=is_blender).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.deform.gaussian_warp.bias)
        nn.init.zeros_(self.deform.gaussian_rotation.bias)
        nn.init.zeros_(self.deform.gaussian_scaling.bias)
        nn.init.normal_(self.deform.gaussian_warp.weight, mean=0, std=1e-6)
        nn.init.normal_(self.deform.gaussian_rotation.weight, mean=0, std=1e-6)
        nn.init.normal_(self.deform.gaussian_scaling.weight, mean=0, std=1e-6)

    def step(self, xyz, d_xyz, d_rot, time_emb):
        if isinstance(d_xyz, torch.Tensor) and isinstance(d_rot, torch.Tensor) and d_rot.shape[-1] == 3:
            xyz = SO3.exp(d_rot).act(xyz) + d_xyz
        else:
            xyz = xyz + d_xyz
        return self.deform(xyz.detach(), time_emb)

    def train_setting(self, training_args):
        params = [{
            'params': list(self.deform.parameters()),
            'lr': training_args.position_lr_init * self.spatial_lr_scale,
            "name": "deform"
        }]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "fine_deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "fine_deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "fine_deform/iteration_{}/deform.pth".format(loaded_iter))
        self.deform.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
