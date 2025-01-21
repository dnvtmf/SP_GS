import os
from typing import Mapping, Any

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from lietorch import SO3
from scene.gaussian_model import change_optimizer
from utils.general_utils import get_expon_lr_func
from utils.sampling import furthest_point_sampling
from utils.system_utils import searchForMaxIteration
from utils.time_utils import SuperpointDeformationNetwork

DIM_ROTATION = 3


def get_superpoint_features(value: Tensor, neighbor: Tensor, G: Tensor, num_sp: int):
    """ value_sp[j] = 1 / w[j] sum_{i=0}^{N} [j in neighbor[i]] G[i, j] value[i]
    w[j] = sum_{i=0}^{N} [j in neighbor[i]] G[i, j]

    Args:
        G: shape [N, K]
        neighbor: shape: [N, K] The indices of K-nearest superpoints for each point
        value: [N, C]
        num_sp: The number of superpoints
    Returns:
        Tensor: the value for superpoints, shape: [num_sp, C]
    """
    C = value.shape[-1]
    assert 0 <= neighbor.min() and neighbor.max() < num_sp
    value_sp = value.new_zeros([num_sp, C])
    value_sp = torch.scatter_reduce(
        value_sp,
        dim=0,
        index=neighbor[:, :, None].repeat(1, 1, C).view(-1, C),
        src=(value[:, None, :] * G[:, :, None]).view(-1, C),
        reduce='sum'
    )
    w = value.new_zeros([num_sp]).scatter_reduce_(dim=0, index=neighbor.view(-1), src=G.view(-1), reduce='sum')
    return value_sp / w[:, None].clamp_min(1e-5)


class SuperpointModel(nn.Module):
    p2sp: Tensor
    sp_xyz: Tensor
    sp_delta_r: Tensor
    sp_delta_t: Tensor
    is_sp_init: Tensor
    train_times: Tensor

    def __init__(self, num_points, train_times=None, num_superpoints=300, num_knn=5):
        super(SuperpointModel, self).__init__()
        self.num_superpoints = num_superpoints
        self.num_knn = num_knn
        self.num_frames = 0 if train_times is None else len(train_times)

        self.A = nn.Parameter(torch.empty(num_points, num_superpoints))
        self.register_buffer('p2sp', torch.zeros(num_points, dtype=torch.long))
        self.register_buffer('sp_xyz', torch.zeros(num_superpoints, 3))
        self.register_buffer('sp_delta_r', torch.zeros(self.num_frames, num_superpoints, DIM_ROTATION))
        self.register_buffer('sp_delta_t', torch.zeros(self.num_frames, num_superpoints, 3))
        self.register_buffer('train_times', train_times)

        self.register_buffer('is_sp_init', torch.tensor(False))

    @property
    def num_points(self):
        return self.A.shape[0]

    @torch.no_grad()
    def init_superpoints(self, points):
        if self.is_sp_init:
            return
        print('Initializing superpoints...')
        sp_idx = furthest_point_sampling(points, self.num_superpoints)
        self.sp_xyz[:].data.copy_(points[sp_idx])
        _, p2sp = torch.topk(torch.cdist(points, self.sp_xyz), k=1, largest=False)
        self.p2sp = p2sp.view(-1)
        self.A.data.copy_(F.one_hot(self.p2sp, self.num_superpoints).float() * 0.9 + 0.1)
        self.is_sp_init = self.is_sp_init.new_tensor(True)

    def interp_deform(self, t: Tensor):
        time_id = torch.searchsorted(self.train_times, t).item()
        if time_id == self.num_frames:
            t1_idx, t2_idx = self.num_frames - 2, self.num_frames - 1
        elif time_id == 0:
            t1_idx, t2_idx = time_id, time_id + 1
        else:
            t1_idx, t2_idx = time_id - 1, time_id
        t1, t2 = self.train_times[t1_idx], self.train_times[t2_idx]
        w = (t - t1) / (t2 - t1)
        sp_delta_t = torch.lerp(self.sp_delta_t[t1_idx], self.sp_delta_t[t2_idx], w)
        sp_delta_r = torch.lerp(self.sp_delta_r[t1_idx], self.sp_delta_r[t2_idx], w)
        return sp_delta_t, sp_delta_r

    def forward(self, sp_delta_t: Tensor, sp_delta_r: Tensor):
        return sp_delta_t[self.p2sp], sp_delta_r[self.p2sp]

    def calc_association_matrix(self, points_c: Tensor):
        with torch.no_grad():
            dist = torch.cdist(points_c, self.sp_xyz)
            _, neighbor = torch.topk(dist, k=self.num_knn, dim=-1, largest=False)  # [N, K]
        G = torch.gather(self.A, dim=1, index=neighbor).softmax(dim=-1)  # [N, K]
        return G, neighbor

    def property_reconstruct_loss(self, G: Tensor, neighbor: Tensor, attr: Tensor, sp_attr: Tensor = None):
        # G: [N, K], neighbor: [N, K], attr: [N, C]
        if sp_attr is None:
            sp_attr = get_superpoint_features(attr, neighbor, G, self.num_superpoints)  # [M, C]
        re_attr = torch.sum(G[:, :, None] * sp_attr[neighbor], dim=1)  # [N, C]
        return F.mse_loss(re_attr, attr)
        # re_sp_attr = get_superpoint_features(attr, neighbor, G, self.num_superpoints)  # [M, C]
        # re_attr = torch.sum(G[:, :, None] * sp_attr[neighbor], dim=1)  # [N, C]
        # return F.mse_loss(re_attr, attr) + F.mse_loss(re_sp_attr, sp_attr)

    def loss(self, points_c: Tensor, points_t: Tensor, delta_r: Tensor, delta_t: Tensor, loss_aux, t=None):
        points_c = points_c.detach()
        points_t = points_t.detach()
        sp_delta_t, sp_delta_r = loss_aux
        if DIM_ROTATION == 3:
            sp_xyz_t = SO3.exp(sp_delta_r).act(self.sp_xyz) + sp_delta_t
        else:
            sp_xyz_t = self.sp_xyz + sp_delta_t
        G, neighbor = self.calc_association_matrix(points_c)
        losses = {
            're_xyz': self.property_reconstruct_loss(G, neighbor, points_t, sp_xyz_t),
            're_rot': self.property_reconstruct_loss(G, neighbor, delta_r, sp_delta_r),
            're_off': self.property_reconstruct_loss(G, neighbor, delta_t, sp_delta_t),
        }
        if t is not None:
            with torch.no_grad():
                tid = torch.nonzero(self.train_times.eq(t))
                assert tid.numel() == 1
                tid = tid.item()
                self.sp_xyz = get_superpoint_features(points_c, neighbor, G, self.num_superpoints)
                self.p2sp = torch.gather(neighbor, -1, G.argmax(dim=-1, keepdim=True))[:, 0]
                self.sp_delta_r[tid] = sp_delta_r
                self.sp_delta_t[tid] = sp_delta_t
        return losses

    def load_state_dict(self, state_dict: Mapping[str, Any], **kwargs):
        self.num_frames = state_dict['train_times'].shape[0]
        num_points, self.num_superpoints = state_dict['A'].shape
        device = self.A.device
        self.A = nn.Parameter(torch.empty(num_points, self.num_superpoints, device=device))
        self.p2sp = torch.zeros(num_points, device=device, dtype=torch.long)
        self.sp_xyz = torch.zeros(self.num_superpoints, 3, device=device)
        self.sp_delta_r = torch.zeros(self.num_frames, self.num_superpoints, DIM_ROTATION, device=device)
        self.sp_delta_t = torch.zeros(self.num_frames, self.num_superpoints, 3, device=device)
        self.train_times = torch.zeros(self.num_frames, device=device)
        super().load_state_dict(state_dict, **kwargs)

    def extra_repr(self) -> str:
        return (f"num_points={self.num_points}, num_superpoints={self.num_superpoints}, "
                f"num_frames={self.num_frames}, knn={self.num_knn}")


class DeformModel:
    def __init__(self, num_points=0, train_times=None, num_superpoints=300, num_knn=5, sp_net_large=True):
        if sp_net_large:
            self.sp_deform = SuperpointDeformationNetwork(D=8, W=256, skips=[4], dim_rot=DIM_ROTATION).cuda()
        else:
            self.sp_deform = SuperpointDeformationNetwork(dim_rot=DIM_ROTATION).cuda()
        self.sp_model = SuperpointModel(num_points, train_times, num_superpoints, num_knn).cuda()
        self.optimizer = None
        self.spatial_lr_scale = 5

    def step(self, xyz, time_emb, use_mlp=True):
        if use_mlp:
            self.sp_model.init_superpoints(xyz)
            sp_delta_xyz, sp_delta_r = self.sp_deform(self.sp_model.sp_xyz, time_emb)
        else:
            sp_delta_xyz, sp_delta_r = self.sp_model.interp_deform(time_emb)
        delta_xyz, delta_r = self.sp_model(sp_delta_xyz, sp_delta_r)
        return (delta_xyz, delta_r, 0), (sp_delta_xyz, sp_delta_r)

    def train_setting(self, training_args):
        gp = [{
            'params': list(self.sp_deform.parameters()),
            'lr': training_args.deform_lr_init,
            "name": "deform"
        }, {
            'params': list(self.sp_model.parameters()),
            'lr': training_args.A_lr_init,
            'name': 'A',
        }]
        self.optimizer = torch.optim.Adam(gp, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=training_args.deform_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps
        )
        self.A_scheduler_args = get_expon_lr_func(
            lr_init=training_args.A_lr_init,
            lr_final=training_args.position_lr_final,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps
        )

    def save_weights(self, model_path, iteration):
        self.update_all_deforms()
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save([self.sp_deform.state_dict(), self.sp_model.state_dict()],
            os.path.join(out_weights_path, 'deform.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "deform/iteration_{}/deform.pth".format(loaded_iter))
        pth = torch.load(weights_path)
        self.sp_deform.load_state_dict(pth[0])
        self.sp_model.load_state_dict(pth[1])

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
            elif param_group["name"] == "A":
                lr = self.A_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def prune_points(self, mask):
        valid_points_mask = ~mask
        self.sp_model.A = change_optimizer(self.optimizer, valid_points_mask, ['A'], op='prune')['A']
        self.sp_model.p2sp = self.sp_model.p2sp[valid_points_mask]

    def clone_points(self, mask):
        new_A = change_optimizer(self.optimizer, {'A': self.sp_model.A[mask]}, op='concat')['A']
        new_p2sp = torch.cat([self.sp_model.p2sp, self.sp_model.p2sp[mask]], dim=0)
        self.sp_model.A = new_A
        self.sp_model.p2sp = new_p2sp

    def split_points(self, mask, N=2):
        new_A = self.sp_model.A[mask].repeat(N, 1)
        self.sp_model.A = change_optimizer(self.optimizer, {'A': new_A}, op='concat')['A']
        new_p2sp = self.sp_model.p2sp[mask].repeat(N)
        self.sp_model.p2sp = torch.cat([self.sp_model.p2sp, new_p2sp], dim=0)

    @torch.no_grad()
    def update_all_deforms(self):
        for i, t in enumerate(self.sp_model.train_times):
            sp_delta_xyz, sp_delta_r = self.sp_deform(self.sp_model.sp_xyz, t)
            self.sp_model.sp_delta_r[i] = sp_delta_r
            self.sp_model.sp_delta_t[i] = sp_delta_xyz
        print("updte all defromations for superpoints")
