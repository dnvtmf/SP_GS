import argparse
import math
from pathlib import Path
from typing import Dict, List

import imageio
import cv2
import dearpygui.dearpygui as dpg
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import seaborn as sns

from utils.general_utils import inverse_sigmoid
from utils.system_utils import searchForMaxIteration
from scene.gaussian_model import GaussianModel
from scene.deform_model import SuperpointModel
from utils.sh_utils import eval_sh
from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer
from utils import ops_3d
from utils.gui_viewer import Viewer3D


def get_colors(num_colors=8, mode=0, return_numpy=False, channels=3, shuffle=True, **kwargs):
    if mode == 0:
        colors = sns.color_palette("hls", num_colors, **kwargs)
    elif mode == 1:
        colors = sns.color_palette('Blues', num_colors)
    else:
        colors = sns.color_palette(n_colors=num_colors, **kwargs)
    colors = np.array(colors)
    if channels == 4:
        colors = np.concatenate([colors, np.ones_like(colors[:, :1])], axis=-1)
    if shuffle:
        np.random.shuffle(colors)
    return colors if return_numpy else torch.from_numpy(colors)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def R_to_quaternion(R: Tensor):
    w = 0.5 * torch.sqrt((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] + 1).clamp_min(1e-10))  # 避免数值不稳定
    w_ = 0.25 / w
    x = (R[..., 2, 1] - R[..., 1, 2]) * w_
    y = (R[..., 0, 2] - R[..., 2, 0]) * w_
    z = (R[..., 1, 0] - R[..., 0, 1]) * w_
    return F.normalize(torch.stack([w, x, y, z], dim=-1), dim=-1)


# noinspection PyArgumentList
class SP_GS_GUI:
    def __init__(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        # self.net, self.db = self.build_dataset_and_net()
        self.image_size = (800, 800)
        self.camera_index = 0
        self.mean_point_scale = 0
        self.sp_colors = None
        self.now_Tw2c = None

        self.device = torch.device('cuda')

        self.now_points = None
        self.saved_video = []

        dpg.create_context()
        dpg.create_viewport(
            title='Superpoint Gaussian Splatting Edit',
            width=self.image_size[0],
            height=self.image_size[1],
        )

        self.is_vary_time = False
        self.is_vary_view = False
        self.is_save_video = False
        self.is_vary_animate = False
        self.is_auto_save_video = 0
        self.auto_save_video_t = 0.
        self.M = 0
        self.tree_parent = torch.full((0,), self.M, dtype=torch.int, device=self.device)
        self.tree_children: Dict[int, List[int]] = {-1: []}
        self.tree_T = torch.zeros(0, 4, 4, device=self.device)
        self.tree_scale = torch.zeros(0, 3, device=self.device)
        self.tree_mask = torch.ones(0, dtype=torch.bool, device=self.device)
        self.tree_center = torch.zeros(0, 3, device=self.device)
        self.num_object = 0
        self.gs = GaussianModel(3)
        self.sp = []  # type: List[SuperpointModel]
        self.p2sp = torch.zeros(0, device=self.device)

        dpg.push_container_stack(dpg.add_window(tag='Primary Window'))
        self.viewer = Viewer3D(self.rendering, size=self.image_size, no_resize=False, no_move=True)
        with (dpg.window(tag='control', label='FPS:', collapsed=False, no_close=True, width=256, height=800)):
            self.build_gui_add_model()
            with dpg.collapsing_header(label='camera', default_open=False):
                self.build_gui_camera()
            with dpg.collapsing_header(label='rendering', default_open=True):
                self.build_gui_render()
            with dpg.collapsing_header(label='display', default_open=True):
                self.build_gui_display()
            with dpg.collapsing_header(label='edit', tag='gui_edit', default_open=True):
                self.build_gui_edit()
        dpg.pop_container_stack()
        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(callback=self.viewer.callback_mouse_drag)
            dpg.add_mouse_wheel_handler(callback=self.viewer.callback_mouse_wheel)
            dpg.add_mouse_release_handler(callback=self.viewer.callback_mouse_release)
            # dpg.add_mouse_wheel_handler(callback=self.callback_mouse_wheel)
            dpg.add_mouse_move_handler(callback=self.callback_mouse_hover)
            dpg.add_mouse_click_handler(callback=self.callback_mouse_click)
            # dpg.add_key_press_handler(callback=self.callback_keypress)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        # dpg.set_primary_window(self.viewer.win_tag, True)
        dpg.set_primary_window("Primary Window", True)
        # dpg.start_dearpygui()

    @property
    def sp_index(self):
        return dpg.get_value('sp_idx')

    @sp_index.setter
    def sp_index(self, c):
        dpg.set_value('sp_idx', c)
        if c >= 0:
            dpg.set_value('cx', self.tree_center[c, 0].item())
            dpg.set_value('cy', self.tree_center[c, 1].item())
            dpg.set_value('cz', self.tree_center[c, 2].item())

    def build_gui_add_model(self):
        def load_model(sender, app_data):
            path = Path(app_data['file_path_name'])
            if path.joinpath('point_cloud').is_dir():
                max_iter = searchForMaxIteration(path.joinpath('point_cloud'))
                ply_path = path.joinpath('point_cloud', f'iteration_{max_iter}', 'point_cloud.ply')
                pth_path = path.joinpath('deform', f'iteration_{max_iter}', 'deform.pth')
            elif path.suffix == '.ply':
                ply_path = path
                pth_path = list(ply_path.parts)
                pth_path[-1] = 'deform.pth'
                pth_path[-3] = 'deform'
                pth_path = Path(*pth_path)
            elif path.suffix == '.pth':
                pth_path = path
                ply_path = list(path.parts)
                ply_path[-1] = 'point_cloud.ply'
                ply_path[-3] = 'point_cloud'
                ply_path = Path(*pth_path)
            else:
                print('Not find a model')
                return
            if not (ply_path.is_file() and pth_path.is_file()):
                print(f'{pth_path} or {ply_path} is invalied')
                return
            gs = GaussianModel(3)
            gs.load_ply(ply_path)
            pth = torch.load(pth_path, map_location='cpu')
            sp_model = SuperpointModel(300).to(self.device)
            sp_model.load_state_dict(pth[1])
            # print(sp_model.train_times)
            if self.num_object == 0:
                self.gs = gs
                self.sp = [sp_model]
                self.p2sp = sp_model.p2sp
                self.M = sp_model.num_superpoints
                self.tree_parent = torch.full((self.M + 1,), self.M, device=self.device, dtype=torch.int)
                self.tree_parent[-1] = -1
                self.tree_children = {self.M: list(range(self.M)), -1: [self.M]}
                self.tree_T = torch.eye(4, device=self.device).expand(self.M + 1, 4, 4).contiguous()
                self.tree_scale = torch.ones(self.M + 1, device=self.device)
                self.tree_center = torch.zeros(self.M + 1, 3, device=self.device)
                self.tree_mask = torch.ones(self.M + 1, dtype=torch.bool, device=self.device)
                self.scene_gui_delete(-1)
                self.mean_point_scale = self.gs.get_scaling.mean()
                now = self.M
            else:
                self.gs._xyz = torch.cat([self.gs._xyz, gs._xyz], dim=0)
                self.gs._scaling = torch.cat([self.gs._scaling, gs._scaling], dim=0)
                self.gs._rotation = torch.cat([self.gs._rotation, gs._rotation], dim=0)
                self.gs._features_dc = torch.cat([self.gs._features_dc, gs._features_dc], dim=0)
                self.gs._features_rest = torch.cat([self.gs._features_rest, gs._features_rest], dim=0)
                self.gs._opacity = torch.cat([self.gs._opacity, gs._opacity], dim=0)

                self.sp.append(sp_model)
                self.p2sp = torch.cat([self.p2sp, sp_model.p2sp + self.M], dim=0)
                N = sp_model.num_superpoints
                self.tree_parent = torch.where(self.tree_parent >= self.M, self.tree_parent + N, self.tree_parent)
                for k, child in self.tree_children.items():
                    self.scene_gui_delete(k)
                    self.tree_children[k] = [x + N if x >= self.M else x for x in child]
                self.tree_children = {k + N if k >= self.M else k: v for k, v in self.tree_children.items()}
                now = len(self.tree_parent) + N
                self.tree_parent = torch.cat([
                    self.tree_parent[:self.M], self.tree_parent.new_full((N,), now),
                    self.tree_parent[self.M:], self.tree_parent.new_tensor([-1])]
                )
                self.tree_children[now] = list(range(self.M, self.M + N))
                self.tree_children[-1].append(now)
                self.tree_parent[now] = -1
                eye = torch.eye(4, device=self.tree_T.device)[None]
                self.tree_T = torch.cat([self.tree_T[:self.M], eye.expand(N, 4, 4), self.tree_T[self.M:], eye], dim=0)
                self.tree_scale = torch.cat([self.tree_scale[:self.M], self.tree_scale.new_ones(N),
                                             self.tree_scale[self.M:], self.tree_scale.new_ones(1)], dim=0)
                self.tree_center = torch.cat([self.tree_center[:self.M], self.tree_center.new_zeros(N, 3),
                                              self.tree_center[self.M:], self.tree_center.new_zeros(1, 3)], dim=0)
                self.tree_mask = torch.cat([
                    self.tree_mask[:self.M], self.tree_mask.new_ones(N), self.tree_mask[self.M:],
                    self.tree_mask.new_ones(1)
                ])
                self.M += N
            self.sp_colors = get_colors(self.M).to(self.device).float()
            self.tree_T[now] = ops_3d.translate(-gs.get_xyz.mean(dim=0))
            self.num_object += 1
            print('add model from:',
                app_data['file_path_name'],
                't:',
                self.sp[-1].train_times.min().item(),
                self.sp[-1].train_times.max().item(),
            )
            dpg.configure_item('sp_idx', max_value=len(self.tree_parent))
            dpg.push_container_stack('gui_edit')
            self.scene_gui_build(-1)
            dpg.pop_container_stack()
            self.viewer.set_need_update()

        with dpg.file_dialog(
            # directory_selector=True,
            show=False,
            callback=load_model,
            id="load_model_dialog_id",
            default_filename='',
            default_path='output/',
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".ply", color=(150, 255, 150, 255))
            dpg.add_file_extension(".pth", color=(255, 150, 150, 255))
            dpg.add_file_extension(".*")

        def add_model():
            dpg.show_item('load_model_dialog_id')
            self.viewer.set_need_update()

        def save_edited_model(sender, app_data):
            path = Path(app_data['file_path_name'])
            data = {
                'sp': self.sp,
                'tree_parent': self.tree_parent.contiguous(),
                'tree_children': self.tree_children,
                'tree_T': self.tree_T.contiguous(),
                'tree_scale': self.tree_scale.contiguous(),
                'tree_mask': self.tree_mask.contiguous(),
                'tree_center': self.tree_center.contiguous(),
                'eye': self.viewer.eye,
                'at': self.viewer.at,
                'up': self.viewer.up,
                'fovy': self.viewer.fovy,
            }
            self.gs.save_ply(path.with_suffix('.ply'))
            torch.save(data, path)
            print('save edit model to', path)

        def load_edited_model(sender, app_data):
            for c in self.tree_children.keys():
                self.scene_gui_delete(c)
            path = Path(app_data['file_path_name'])
            self.gs.load_ply(path.with_suffix('.ply'))
            data = torch.load(path, map_location='cpu')
            self.sp = [sp.to(self.device) for sp in data['sp']]
            self.num_object = len(self.sp)
            self.tree_mask = data['tree_mask'].to(self.device)
            self.tree_parent = data['tree_parent'].to(self.device)
            self.tree_children = data['tree_children']
            self.tree_T = data['tree_T'].requires_grad_(False).to(self.device)
            self.tree_scale = data['tree_scale'].to(self.device).requires_grad_(False)
            self.tree_center = data['tree_center'].to(self.device)
            p2sp = []
            M = 0
            for i in range(self.num_object):
                p2sp.append(self.sp[i].p2sp + M)
                M += self.sp[i].sp_delta_t.shape[1]
            self.p2sp = torch.cat(p2sp)
            self.M = M
            self.sp_colors = get_colors(self.M).to(self.device).float()
            dpg.configure_item('sp_idx', max_value=len(self.tree_parent))
            dpg.push_container_stack('gui_edit')
            self.scene_gui_build(-1)
            dpg.pop_container_stack()
            for c in self.tree_children.keys():
                dpg.set_value(f'node_show_{c}', self.tree_mask[c].item())
            self.viewer.set_pose(data['eye'].cpu(), data['at'].cpu(), data['up'].cpu())
            self.viewer.set_fovy(math.degrees(data['fovy']))
            self.viewer.set_need_update()
            print('loaded edit model from', path)

        with dpg.group(horizontal=True):
            dpg.add_button(label='Add Model', callback=add_model)

            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=save_edited_model,
                id="save_edit_dialog_id",
                default_filename='edit',
                default_path='output/edit',
                width=700,
                height=400,
            ):
                dpg.add_file_extension(".sp_gs", color=(150, 255, 150, 255))
                dpg.add_file_extension(".*")
            dpg.add_button(label='Save', callback=lambda: dpg.show_item('save_edit_dialog_id'))

            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=load_edited_model,
                id="load_edit_dialog_id",
                default_filename='edit',
                default_path='output/edit',
                width=700,
                height=400,
            ):
                dpg.add_file_extension(".sp_gs", color=(150, 255, 150, 255))
                dpg.add_file_extension(".*")
            dpg.add_button(label='Load', callback=lambda: dpg.show_item('load_edit_dialog_id'))

    def build_gui_camera(self):
        # dpg.add_text(tag='fps')
        with dpg.group(horizontal=True):
            dpg.add_text('fovy')
            dpg.add_slider_float(
                min_value=15.,
                max_value=180.,
                default_value=math.degrees(self.viewer.fovy),
                callback=lambda *args: self.viewer.set_fovy(dpg.get_value('set_fovy')),
                tag='set_fovy'
            )
        with dpg.group():
            item_width = 50
            with dpg.group(horizontal=True):
                dpg.add_text('eye')
                dpg.add_input_float(tag='eye_x', step=0, width=item_width)
                dpg.add_input_float(tag='eye_y', step=0, width=item_width)
                dpg.add_input_float(tag='eye_z', step=0, width=item_width)
            with dpg.group(horizontal=True):
                dpg.add_text('at ')
                dpg.add_input_float(tag='at_x', step=0, width=item_width)
                dpg.add_input_float(tag='at_y', step=0, width=item_width)
                dpg.add_input_float(tag='at_z', step=0, width=item_width)

            def change_eye(*args):
                print('change camera position', args)
                self.viewer.eye = self.viewer.eye.new_tensor([dpg.get_value(item) for item in
                                                              ['eye_x', 'eye_y', 'eye_z']])
                self.viewer.at = self.viewer.at.new_tensor([dpg.get_value(item) for item in
                                                            ['at_x', 'at_y', 'at_z']])
                self.viewer.need_update = True

            def to_camera_pos(x=1, y=0, z=0):
                def callback():
                    r = (self.viewer.eye - self.viewer.at).norm()
                    eye = self.viewer.eye.new_tensor([x, y, z])
                    self.viewer.eye = eye / eye.norm(keepdim=True) * r + self.viewer.at
                    self.viewer.set_need_update()

                return callback

            with dpg.group(horizontal=True):
                dpg.add_button(label='change', callback=change_eye)
                dpg.add_button(label='+X', callback=to_camera_pos(1, 0, 0))
                dpg.add_button(label='-X', callback=to_camera_pos(-1, 0, 0))
                dpg.add_button(label='+Y', callback=to_camera_pos(0, 1, 0))
                dpg.add_button(label='-Y', callback=to_camera_pos(0, -1, 0))
                dpg.add_button(label='+Z', callback=to_camera_pos(0, 0, 1))
                dpg.add_button(label='-Z', callback=to_camera_pos(0, 0, -1))
            dpg.add_input_float(label='axis len', tag='axis_len',
                min_value=0.1, max_value=10, default_value=0.5, callback=self.viewer.set_need_update)
            dpg.add_slider_int(
                min_value=1,
                max_value=10,
                default_value=1,
                tag='axis_size',
                label='axis size',
                callback=self.viewer.set_need_update)

    def build_gui_render(self):
        with dpg.group(horizontal=True):
            dpg.add_slider_float(label='t', tag='time', max_value=1.0, callback=self.viewer.set_need_update)

            def vary_time():
                self.is_vary_time = not self.is_vary_time

            dpg.add_button(label='A', callback=vary_time)

        def set_rotate_index_limit():
            dpg.set_value('rotate_t', 0)
            self.viewer.set_need_update()

        with dpg.group(horizontal=True):
            dpg.add_text('Rotate Obj: auto')
            dpg.add_checkbox(tag='rotate_auto', callback=self.viewer.set_need_update)
            dpg.add_button(tag='roate_reset', label='R', callback=set_rotate_index_limit)
        with dpg.group(horizontal=True):
            dpg.add_slider_float(tag='rotate_t', callback=self.viewer.set_need_update, width=100, max_value=1)
            dpg.add_text('/')

            dpg.add_input_int(tag='rotate_speed',
                step=0,
                default_value=360,
                min_value=10,
                min_clamped=True,
                width=50,
                callback=set_rotate_index_limit)

    def build_gui_display(self):
        with dpg.group(horizontal=True):
            dpg.add_text('show')
            dpg.add_text('size:')
            dpg.add_slider_float(tag='point_size',
                min_value=0,
                max_value=2.0,
                default_value=1.0,
                width=100,
                callback=self.viewer.set_need_update
            )
        with dpg.group(horizontal=True):
            dpg.add_text('points')
            dpg.add_checkbox(tag='show_points', callback=self.viewer.set_need_update)
            dpg.add_text('superpoints')
            dpg.add_checkbox(tag='show_superpoints', callback=self.viewer.set_need_update)
            dpg.add_text('p2sp')
            dpg.add_checkbox(tag='show_p2sp', callback=self.viewer.set_need_update)

        with dpg.group(horizontal=True):
            def save_image(sender, app_data):
                imageio.v3.imwrite(app_data['file_path_name'], ops_3d.as_np_image(self.viewer.data))
                print('save image to', app_data['file_path_name'])

            with dpg.file_dialog(directory_selector=False,
                show=False,
                callback=save_image,
                id="save_file_dialog_id",
                default_filename='edit',
                width=700,
                height=400):
                dpg.add_file_extension("images{.jpg,.jpeg,.png}", color=(150, 150, 255, 255))
                dpg.add_file_extension(".jpg", color=(150, 255, 150, 255))
                dpg.add_file_extension(".png", color=(255, 150, 150, 255))
                dpg.add_file_extension(".*")

            dpg.add_button(label='save_image', callback=lambda: dpg.show_item('save_file_dialog_id'))

        with dpg.group(horizontal=True):
            def save_video(sender, app_data):
                videos = np.stack(self.saved_video, axis=0)
                save_path = Path(app_data['file_path_name'])
                # utils.save_mp4(save_path, videos, fps=dpg.get_value('fps_video'))
                imageio.v3.imwrite(save_path, videos, fps=dpg.get_value('fps_video'), quality=8)
                self.saved_video = []
                dpg.configure_item('save_video', label='start')
                print(f"save videos {videos.shape} to {save_path}")

            with dpg.file_dialog(directory_selector=False,
                show=False,
                callback=save_video,
                id="save_video_dialog_id",
                default_filename='edit',
                default_path='output/edit',
                width=700,
                height=400):
                dpg.add_file_extension(".mp4", color=(150, 255, 150, 255))

            def save_video_callback():
                if not self.is_save_video:
                    self.is_save_video = True
                    self.saved_videos = []
                    dpg.configure_item('save_video', label='save(0)')
                else:
                    self.is_save_video = False
                    dpg.show_item('save_video_dialog_id')
                self.viewer.set_need_update()

            dpg.add_text('save_video')
            # dpg.add_checkbox(tag='save_video', callback=save_video_callback)
            dpg.add_button(label='start', tag='save_video', callback=save_video_callback)

            def clear_saved():
                self.saved_video.clear()
                if self.is_save_video:
                    dpg.configure_item('save_video', label='save(0)')
                else:
                    dpg.configure_item('save_video', label='start')

            dpg.add_button(label='clear', callback=clear_saved)

            def auto_save_video():
                self.is_auto_save_video = 1
                self.saved_video = []
                self.is_save_video = True
                dpg.configure_item('save_video', label='save(0)')

            dpg.add_button(label='auto', tag='auto_save_video', callback=auto_save_video)
        with dpg.group(horizontal=True):
            dpg.add_text('frames')
            dpg.add_input_int(default_value=120, tag='n_video', width=100, step=5)
            dpg.add_text('FPS')
            dpg.add_input_float(default_value=30, tag='fps_video', width=50, step=0)

    def get_edit_sp_transform(self, scale=1.):
        rx = dpg.get_value('rot x') * dpg.get_value('rot x_max') / 180. * torch.pi * scale
        ry = dpg.get_value('rot y') * dpg.get_value('rot y_max') / 180. * torch.pi * scale
        rz = dpg.get_value('rot z') * dpg.get_value('rot z_max') / 180. * torch.pi * scale
        tx = dpg.get_value('off x') * dpg.get_value('off x_max') * scale
        ty = dpg.get_value('off y') * dpg.get_value('off y_max') * scale
        tz = dpg.get_value('off z') * dpg.get_value('off z_max') * scale
        scale = dpg.get_value('scale_max') ** dpg.get_value('scale')
        cx, cy, cz = dpg.get_value('cx'), dpg.get_value('cy'), dpg.get_value('cz')
        T = ops_3d.translate(-cx, -cy, -cz, device=self.device)
        T = ops_3d.rotate(rx, ry, rz, device=self.device) @ T
        T = ops_3d.scale(scale, device=self.device) @ T
        T = ops_3d.translate(tx + cx, ty + cy, tz + cz, device=self.device) @ T
        return T, scale

    def build_gui_edit(self):
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label='enable', tag='edit', callback=self.viewer.set_need_update)

        with dpg.group(horizontal=True):
            dpg.add_text('now:')
            dpg.add_input_int(tag='sp_idx',
                default_value=-1,
                min_value=-1,
                max_value=0,
                min_clamped=True,
                max_clamped=True,
                callback=self.viewer.set_need_update,
                width=100,
            )

            def apply_transform():
                sp_idx = dpg.get_value('sp_idx')
                if sp_idx < 0:
                    return
                T, scale = self.get_edit_sp_transform()
                self.tree_T[sp_idx] = self.tree_T[sp_idx] @ T
                self.tree_scale[sp_idx] = self.tree_scale[sp_idx] * scale
                for name in ['rot x', 'rot y', 'rot z', 'off x', 'off y', 'off z', 'scale']:
                    dpg.set_value(name, 0.)
                self.viewer.set_need_update()

            dpg.add_button(label='apply', callback=apply_transform)

            def reset_transform():
                sp_idx = dpg.get_value('sp_idx')
                if sp_idx >= 0:
                    self.tree_T[sp_idx] = torch.eye(4, device=self.device)
                self.viewer.set_need_update()

            dpg.add_button(label='reset', callback=reset_transform)
        with dpg.group(horizontal=True):
            def switch_animate():
                self.is_vary_animate = not self.is_vary_animate

            dpg.add_button(label='Animate')
            dpg.add_slider_float(min_value=0, max_value=1, width=100, default_value=1,
                tag='animate_t', callback=self.viewer.set_need_update)
            dpg.add_input_int(min_value=1, min_clamped=True, default_value=120, tag='animate_speed', step=0, width=40)
            dpg.add_button(label='A', callback=switch_animate)

        with dpg.collapsing_header(label='transform'):
            def reset_callback(name, v, v_max):
                def callback(*args):
                    dpg.set_value(name, v)
                    dpg.set_value(name + '_max', v_max)
                    self.viewer.set_need_update()

                return callback

            for name, v, v_max in [
                ('rot x', 0, 30), ('rot y', 0, 30), ('rot z', 0, 30),
                ('off x', 0, 1), ('off y', 0, 1), ('off z', 0, 1),
                ('scale', 0, 2)
            ]:
                with dpg.group(horizontal=True):
                    dpg.add_text(name)
                    dpg.add_slider_float(
                        tag=name, default_value=v, width=100, min_value=-1, max_value=1,
                        callback=self.viewer.set_need_update
                    )
                    dpg.add_input_float(
                        tag=name + '_max', default_value=v_max, width=50, step=0,
                        callback=self.viewer.set_need_update
                    )
                    dpg.add_button(label='R', callback=reset_callback(name, v, v_max))
            with dpg.group(horizontal=True):
                def chage_center(name_, i):
                    def callback():
                        sp_idx = self.sp_index
                        if sp_idx >= 0:
                            self.tree_center[sp_idx, i] = dpg.get_value(f"c{name_}")

                    return callback

                dpg.add_text('center:')
                for i, name in enumerate(['x', 'y', 'z']):
                    dpg.add_input_float(
                        default_value=0, step=0, tag=f'c{name}', width=50, callback=chage_center(name, i))
        self.scene_gui_build(-1)

    def scene_gui_add(self, x, depth=0, parent: int = 0):
        leafs = [str(c) for c in self.tree_children[x] if c < self.M]
        with dpg.collapsing_header(
            label=('scene' if x < 0 else f'{x}'), tag=f'node_h{x}', indent=depth * 10,
            parent=f"node_h{parent}" if x > 0 else 0, open_on_arrow=True
        ):
            with dpg.group(horizontal=True, tag=f'node_g{x}'):
                def show_hidden_node():
                    self.tree_mask[x] = dpg.get_value(f'node_show_{x}')
                    self.viewer.set_need_update()

                dpg.add_checkbox(tag=f'node_show_{x}', default_value=True, callback=show_hidden_node)

                def add_to_now_set(goal):
                    def cbk():
                        sp_idx = dpg.get_value('sp_idx')
                        if sp_idx < 0 or goal < self.M:
                            return
                        y = sp_idx
                        while y >= 0:
                            if y == goal:
                                return
                            y = self.tree_parent[y].item()
                        y = self.tree_parent[sp_idx].item()
                        self.tree_children[y].remove(sp_idx)
                        self.tree_children[goal].append(sp_idx)
                        self.tree_parent[sp_idx] = goal

                        T0 = self.tree_T[y]
                        s0 = self.tree_scale[y]
                        z = self.tree_parent[y].item()
                        while z >= 0:
                            T0 = self.tree_T[z] @ T0
                            s0 = self.tree_scale[z] * s0
                            z = self.tree_parent[z].item()
                        T1 = self.tree_T[goal]
                        s1 = self.tree_scale[goal]
                        z = self.tree_parent[goal].item()
                        while z >= 0:
                            T1 = self.tree_T[z] @ T1
                            s1 = self.tree_scale[z] * s1
                            z = self.tree_parent[z].item()
                        self.tree_T[sp_idx] = torch.inverse(T1) @ T0 @ self.tree_T[sp_idx]
                        self.tree_scale[sp_idx] = self.tree_scale[sp_idx] * s0 / s1
                        self.scene_gui_update(y)
                        self.scene_gui_update(goal)
                        self.viewer.set_need_update()
                        print(f"move node {sp_idx} to set {goal}")

                    return cbk

                dpg.add_button(label=f'+', tag=f'node_add_{x}', callback=add_to_now_set(x))

                def new_set(goal):
                    def cbk():
                        now = len(self.tree_parent)
                        self.tree_children[now] = []
                        self.tree_children[goal].append(now)
                        self.tree_parent = torch.cat([self.tree_parent, self.tree_parent.new_tensor([goal])])
                        self.tree_parent[now] = goal
                        self.tree_T = torch.cat([self.tree_T, torch.eye(4, device=self.tree_T.device)[None]], dim=0)
                        self.tree_mask = torch.cat([self.tree_mask, torch.ones_like(self.tree_mask[:1])])
                        self.tree_center = torch.cat([self.tree_center, torch.zeros_like(self.tree_center[:1])])
                        self.tree_scale = torch.cat([self.tree_scale, torch.ones_like(self.tree_scale[:1])])
                        depth_ = dpg.get_item_configuration(f"node_h{x}")['indent'] // 10
                        self.scene_gui_add(now, depth_ + 1, goal)
                        self.viewer.set_need_update()
                        print(f"add new set {now} as child of set {goal}")
                        dpg.configure_item('sp_idx', max_value=len(self.tree_parent))

                    return cbk

                dpg.add_button(label='new', tag=f'node_new_{x}', callback=new_set(x))

                def callback():
                    self.sp_index = int(dpg.get_value(f"node_child_{x}"))
                    self.viewer.set_need_update()

                dpg.add_combo(items=leafs, tag=f'node_child_{x}',
                    default_value=leafs[0] if len(leafs) > 0 else '', width=50, callback=callback)

                def del_set(goal):
                    def cbk():
                        if goal < 0:
                            return
                        p = self.tree_parent[goal].item()
                        self.tree_children[p].remove(goal)
                        for c in self.tree_children[goal]:
                            self.tree_parent[c] = p
                            self.tree_children[p].append(c)
                            self.tree_T[c] = self.tree_T[goal] @ self.tree_T[c]
                            self.tree_scale[c] = self.tree_scale[goal] * self.tree_scale[c]
                        self.scene_gui_delete(goal)
                        self.scene_gui_update(p)
                        self.tree_children.pop(goal)
                        self.viewer.set_need_update()
                        print(f"del new set {goal}")

                    return cbk

                dpg.add_button(label='del', tag=f'node_del_{x}', callback=del_set(x), enabled=x >= 0)

    def scene_gui_update(self, x: int):
        leafs = [str(c) for c in self.tree_children[x] if c < self.M]
        dpg.configure_item(f"node_child_{x}", items=leafs)
        depth = dpg.get_item_configuration(f"node_h{x}")['indent'] // 10
        que = [(c, depth + 1) for c in self.tree_children[x] if c >= self.M]
        i = 0
        while i < len(que):
            x, depth = que[i]
            dpg.configure_item(f"node_h{x}", indent=depth * 10)
            for c in self.tree_children[x]:
                if c >= self.M:
                    que.append((c, depth + 1))
            i += 1

    def scene_gui_delete(self, x: int):
        for name in ['node_show_', 'node_new_', 'node_add_', 'node_del_', 'node_child_', 'node_h', 'node_g']:
            dpg.delete_item(f'{name}{x}')
            if dpg.does_alias_exist(f'{name}{x}'):
                dpg.remove_alias(f'{name}{x}')

    def scene_gui_build(self, x: int, p=-1, depth=0):
        if x < 0 or x >= self.M:
            self.scene_gui_add(x, depth=depth, parent=p)
            for c in self.tree_children[x]:
                self.scene_gui_build(c, x, depth + 1)
        return

    def options(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='results/SP_GS/joint/jumpingjacks/best.pth')
        parser.add_argument('-i', '--load', type=str, default='./exps/sp_gs/dnerf.yaml')
        parser.add_argument('-s', '--scene', type=str, default=None)
        parser.add_argument('--split', default='train')
        args = parser.parse_args()
        return args

    def get_colors(self, sh_features, points, camera_center, max_sh_degree=3):
        shs_view = sh_features.transpose(1, 2).view(-1, 3, (max_sh_degree + 1) ** 2)
        dir_pp = (points - camera_center.repeat(sh_features.shape[0], 1))
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(max_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp

    @torch.no_grad()
    def rendering(self, Tw2v, fovy, size):
        if self.num_object == 0:
            return self.viewer.data
        Tw2v = Tw2v.cuda()
        # Tw2v = ops_3d.convert_coord_system(Tw2v, 'opengl', 'colmap')
        Tv2c = ops_3d.perspective_v2(fovy, size=self.image_size).cuda()
        Tv2w = torch.inverse(Tw2v)
        Tw2c = Tv2c @ Tw2v
        self.now_Tw2c = Tw2c
        tanfovx = math.tan(ops_3d.fovx_to_fovy(fovy, size[1] / size[0]) * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        bg_color = torch.zeros(3, device=self.device)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(size[1]),
            image_width=int(size[0]),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.,
            viewmatrix=Tw2v.transpose(-1, -2).contiguous(),
            projmatrix=Tw2c.transpose(-1, -2).contiguous(),
            sh_degree=self.gs.max_sh_degree,
            campos=Tv2w[:3, 3],
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        t = torch.tensor([dpg.get_value('time')]).cuda()

        sp_delta_t, sp_delta_r = [], []
        for sp in self.sp:
            sp_delta_t_i, sp_delta_r_i = sp.interp_deform(t)
            sp_delta_r.append(sp_delta_r_i)
            sp_delta_t.append(sp_delta_t_i)
        sp_delta_t, sp_delta_r = torch.cat(sp_delta_t), torch.cat(sp_delta_r)
        out = {
            'scales': self.gs.get_scaling,
            'opacity': self.gs.get_opacity,
            'points': self.gs.get_xyz + sp_delta_t[self.p2sp],
            'rotations': F.normalize(self.gs.get_rotation + sp_delta_r[self.p2sp], dim=-1),
            'shs': self.gs.get_features,
        }

        rotate_angle = 2.0 * torch.pi * dpg.get_value('rotate_t')
        R = ops_3d.rotate_z(rotate_angle).cuda()
        out['points'] = ops_3d.xfm(out['points'], R)
        out['rotations'] = quaternion_multiply(R_to_quaternion(R)[None], out['rotations'])
        # out['shs'] = rotation_SH(out['shs'], R)

        point_scale = 10 ** ((dpg.get_value('point_size') * 0.5 - 1) * 2)  # [1e-2, 1.]
        if dpg.get_value('show_points'):
            out['scales'] = torch.full_like(out['scales'], self.mean_point_scale * point_scale).float()

        out['colors'] = self.get_colors(out.pop('shs'), out['points'], raster_settings.campos)
        if dpg.get_value('show_p2sp'):
            out['colors'] = self.sp_colors[self.p2sp]

        is_edit = dpg.get_value('edit')
        if is_edit and 0 <= dpg.get_value('sp_idx') < self.M:
            out['colors'] = torch.where(self.p2sp.eq(dpg.get_value('sp_idx'))[:, None],
                out['colors'], out['colors'] * .5)

        M = self.M
        p2sp = self.p2sp
        if dpg.get_value('show_superpoints'):
            sp_xyz = torch.cat([sp.sp_xyz for sp in self.sp], dim=0) + sp_delta_t
            sp_xyz = ops_3d.xfm(sp_xyz, R)
            if dpg.get_value('show_points'):
                self.add_gaussians(out,
                    points=sp_xyz,
                    colors=self.sp_colors,
                    scales=self.mean_point_scale * point_scale * 5.0
                )
                p2sp = torch.cat([self.p2sp, torch.arange(M, device=self.device)], dim=0)
            else:
                out = self.add_gaussians(
                    out,
                    points=sp_xyz,
                    colors=self.sp_colors,
                    scales=self.mean_point_scale * point_scale * 5.,
                    replace=True
                )
                p2sp = torch.arange(M, device=self.device)

        T = self.tree_T.clone()
        scales = self.tree_scale.clone()
        sp_idx = dpg.get_value('sp_idx')
        if sp_idx >= 0:
            T_i, s_i = self.get_edit_sp_transform(dpg.get_value('animate_t'))
            T[sp_idx] = T[sp_idx] @ T_i
            scales[sp_idx] *= s_i
        mask = torch.ones(T.shape[0], dtype=torch.bool, device=self.device)
        p = self.tree_parent.clone()
        p[self.M:] = -1
        while p.ge(0).any():
            m = p >= 0
            T[m] = T[m] @ T[p[m]]
            scales[m] = scales[m] * scales[p[m]]
            mask[m] = torch.logical_and(mask[m], self.tree_mask[p[m]])
            p[m] = self.tree_parent[p[m]]
        mask = mask[p2sp]
        out['opacity'] *= mask.float()[:, None]
        out['points'] = ops_3d.xfm(out['points'][:, None], T[p2sp])[:, 0]
        out['rotations'] = quaternion_multiply(R_to_quaternion(T[p2sp][:, :3, :3]), out['rotations'])
        # out['points'] = out['points'] * scales[p2sp][:, None]
        out['scales'] = out['scales'] * scales[p2sp][:, None]
        # net_out['points'] = ops_3d.xfm(net_out['points'][:, None], T[p2sp])[:, 0]

        images, radii = rasterizer(
            means3D=out['points'],
            means2D=torch.zeros_like(out['points']),
            shs=None,  # self.gs.get_features,
            colors_precomp=out['colors'],
            opacities=out['opacity'],
            scales=out['scales'],
            rotations=F.normalize(out['rotations'], dim=-1),
            cov3D_precomp=None)[:2]

        images = torch.permute(images, (1, 2, 0))

        mask = torch.logical_and(radii > 0, mask)
        points = ops_3d.xfm(out['points'], Tw2c, homo=True)
        # mask = mask | points[:, 0] < -1 | points[:, 0] > 1 | points[:, 1] < -1 | points[: 1] > 1 | points[]
        # z = points[:, -1:]
        # z = torch.where(z.abs() < 1e-5, torch.full_like(z, 1e-5), z)
        points[:, :2] = ((points[:, :2] / points[:, -1:] + 1) * points.new_tensor(size) - 1) * 0.5
        self.now_points = torch.where(mask[:, None], points, points.new_tensor([[-10, -10., -10, -10]]))

        if sp_idx >= 0:
            origin = torch.tensor([dpg.get_value(name) for name in ['cx', 'cy', 'cz']], device=self.device)
            axis_len = dpg.get_value('axis_len')
            axis_size = dpg.get_value('axis_size')
            axis = torch.stack([
                origin,  # O
                origin + origin.new_tensor([axis_len, 0, 0]),  # X
                origin + origin.new_tensor([0, axis_len, 0]),  # Y
                origin + origin.new_tensor([0, 0, axis_len]),  # Z
            ])
            axis = ops_3d.xfm(axis, Tw2c, homo=True)
            axis = ((axis[:, :2] / axis[:, -1:] + 1) * axis.new_tensor(size) - 1) * 0.5
            axis = axis.cpu().numpy().astype(np.int32)
            images = np.ascontiguousarray(ops_3d.as_np_image(images))
            images = cv2.line(images, axis[0], axis[1], [255, 0, 0], axis_size)
            images = cv2.line(images, axis[0], axis[2], [0, 255, 0], axis_size)
            images = cv2.line(images, axis[0], axis[3], [0, 0, 255], axis_size)
        # background = torch.rand_like(images)
        # background: Tensor = None
        # if background is not None:
        #     images = images + (1 - render_out_f['opacity'][..., None]) * background.squeeze(0)
        # if 'images_c' in outputs:
        #     images_c = outputs['images_c']
        #     return torch.cat([images, images_c], dim=2)[0]
        # else:
        #     return images[0]
        return images

    def add_lines(self, net_out, line_p1: Tensor, line_p2: Tensor, line_width=1.0):
        points = (line_p1 + line_p2) / 2
        scales = torch.ones_like(points) * line_width * self.mean_point_scale
        dist = torch.pairwise_distance(line_p1, line_p2)
        scales[:, 0] = dist / 6

        if torch.all(line_p1 == line_p2):
            rotations = None
        else:
            v = line_p2 - points
            rotations = ops_3d.direction_vector_to_quaternion(v.new_tensor([[1, 0, 0]]), v)
            rotations = ops_3d.normalize(rotations)
        return self.add_gaussians(net_out, points, points.new_tensor([0., 1., 0.]), scales, rotations)

    def add_gaussians(self, net_out, points, colors, scales, rotations=None, opacity=None, replace=False):
        P = points.shape[0]
        colors = colors.view(-1, 3).expand(P, 3)
        if isinstance(scales, Tensor):
            scales = scales.view(-1, scales.shape[-1] if scales.ndim > 0 else 1).expand(P, 3)
        else:
            scales = net_out['scales'].new_tensor([scales]).view(-1, 1).expand(P, 3)
        if rotations is None:
            rotations = net_out['rotations'].new_zeros([P, 4])
            rotations[:, -1] = 1.
        else:
            rotations = rotations.view(-1, 4).expand(P, 4)
        if opacity is None:
            opacity = net_out['opacity'].new_ones([P, 1])
        else:
            opacity = opacity.view(-1, 1).expand(P, 1)
        if replace:
            net_out.update({
                'points': points,
                'colors': colors,
                'scales': scales,
                'rotations': rotations,
                'opacity': opacity
            })
        else:
            net_out['points'] = torch.cat([net_out['points'], points], dim=0)
            net_out['colors'] = torch.cat([net_out['colors'], colors], dim=0)
            net_out['scales'] = torch.cat([net_out['scales'], scales], dim=0)
            net_out['rotations'] = torch.cat([net_out['rotations'], rotations], dim=0)
            net_out['opacity'] = torch.cat([net_out['opacity'], opacity], dim=0)
        return net_out

    def run(self):
        last_size = None
        while dpg.is_dearpygui_running():
            dpg.render_dearpygui_frame()
            if self.is_auto_save_video:
                t = self.auto_save_video_t / dpg.get_value('n_video')
                if self.is_vary_time:
                    dpg.set_value('time', t)
                if dpg.get_value('rotate_auto'):
                    dpg.set_value('rotate_t', t)
                if self.is_vary_animate:
                    dpg.set_value('animate_t', t)
                self.auto_save_video_t += 1
                self.viewer.need_update = True
            else:
                if self.is_vary_time:
                    t = dpg.get_value('time')
                    t = t + 0.01
                    if t > 1:
                        t = 0.
                    dpg.set_value('time', t)
                    self.viewer.need_update = True

                if dpg.get_value('rotate_auto'):
                    t = dpg.get_value('rotate_t') + 1 / dpg.get_value('rotate_speed')
                    if t > 1:
                        t -= 1
                    dpg.set_value('rotate_t', t)
                    self.viewer.need_update = True

                if self.is_vary_animate:
                    t = dpg.get_value('animate_t') + 1 / dpg.get_value('animate_speed')
                    if t > 1:
                        t -= 1.
                    dpg.set_value('animate_t', t)
                    self.viewer.need_update = True
            if self.viewer.need_update:
                dpg.set_value('eye_x', self.viewer.eye[0].item())
                dpg.set_value('eye_y', self.viewer.eye[1].item())
                dpg.set_value('eye_z', self.viewer.eye[2].item())
                dpg.set_value('at_x', self.viewer.at[0].item())
                dpg.set_value('at_y', self.viewer.at[1].item())
                dpg.set_value('at_z', self.viewer.at[2].item())
            self.viewer.update()
            now_size = self.viewer.size
            if last_size != now_size:
                dpg.configure_item('control', pos=(dpg.get_item_width(self.viewer.win_tag), 0))
                dpg.set_viewport_width(dpg.get_item_width(self.viewer.win_tag) + dpg.get_item_width('control'))
                dpg.set_viewport_height(dpg.get_item_height(self.viewer.win_tag))
                last_size = now_size
            dpg.configure_item('control', label=f"FPS: {dpg.get_frame_rate()}")
            if self.is_save_video:
                self.saved_video.append(ops_3d.as_np_image(self.viewer.data).copy())
                dpg.configure_item('save_video', label=f"save({len(self.saved_video)})")
                if len(self.saved_video) == dpg.get_value('n_video'):
                    self.is_auto_save_video = False
                    dpg.get_item_callback('save_video')()
        dpg.destroy_context()

    def callback_mouse_click(self, sender, app_data):
        if dpg.is_item_clicked(self.viewer.image_tag):
            x, y = self.viewer.get_mouse_pos()
            if self.now_points is not None:
                if dpg.is_key_down(dpg.mvKey_C):
                    sp_idx = dpg.get_value('sp_idx')
                    if sp_idx < 0 or self.now_Tw2c is None:
                        return
                    center = self.tree_center[sp_idx]
                    center = ops_3d.xfm(center, self.now_Tw2c, homo=True)
                    center[0] = ((x * 2 + 1) / self.viewer.size[0] - 1) * center[-1]
                    center[1] = ((y * 2 + 1) / self.viewer.size[1] - 1) * center[-1]
                    center = ops_3d.xfm(center, self.now_Tw2c.inverse())
                    center = center / center[-1]
                    dpg.set_value('cx', center[0].item())
                    dpg.set_value('cy', center[1].item())
                    dpg.set_value('cz', center[2].item())
                    self.tree_center[sp_idx] = center[:3]
                elif dpg.get_value('edit'):
                    pixel = self.now_points[:, :2].round().int()
                    r = 5
                    mask = (pixel[:, 0] >= x - r) & (pixel[:, 0] <= x + r)
                    mask = mask & (pixel[:, 1] >= y - r) & (pixel[:, 1] <= y + r)
                    if mask.sum() == 0:
                        sp_idx = -1
                    else:
                        sp_idx = self.p2sp[mask][torch.argmin(self.now_points[:, 2][mask])].item()
                    # near_sp = defaultdict(int)
                    # sp_idx = -1
                    # num_sp = 0
                    # for dx in range(-r, r + 1):
                    #     for dy in range(-r, r + 1):
                    #         mask = pixel.eq(pixel.new_tensor([x + dx, y + dy])).all(dim=1)
                    #         if mask.sum() == 0:
                    #             continue
                    #         print(self.now_points[mask])
                    #         idx = self.p2sp[mask][torch.argmax(self.now_points[:, 2][mask])].item()
                    #         near_sp[idx] += 1
                    #         if near_sp[idx] > num_sp:
                    #             num_sp = near_sp[idx]
                    #             sp_idx = idx
                    dpg.set_value('sp_idx', sp_idx)
                self.viewer.set_need_update()
        for c in self.tree_children.keys():
            if dpg.is_item_clicked(f"node_h{c}"):
                self.sp_index = c

    def callback_mouse_hover(self, sender, app_data):
        if dpg.is_item_hovered(self.viewer.image_tag):
            pass

    def callback_keypress(self, sender, app_data):
        pass


def rotation_SH(sh: Tensor, R: Tensor):
    """Reference:
        https://en.wikipedia.org/wiki/Wigner_D-matrix
        https://github.com/andrewwillmott/sh-lib
        http://filmicworlds.com/blog/simple-and-fast-spherical-harmonic-rotation/
    """
    from scipy.spatial.transform import Rotation
    import sphecerix

    # option 1
    Robj = Rotation.from_matrix(R[..., :3, :3].cpu().numpy())
    B, N, _ = sh.shape
    sh = sh.transpose(1, 2).reshape(-1, N)
    new_sh = sh.clone()
    cnt = 0
    i = 0
    while cnt < N:
        D = sphecerix.tesseral_wigner_D(i, Robj)
        D = torch.from_numpy(D).to(sh)
        new_sh[:, cnt:cnt + D.shape[0]] = sh[:, cnt:cnt + D.shape[0]] @ D.T
        cnt += D.shape[0]
        i += 1

    # option 2
    # from e3nn import o3
    # rot_angles = o3._rotation.matrix_to_angles(R)
    # D_2 = o3.wigner_D(2, rot_angles[0], rot_angles[1], rot_angles[2])
    #
    # Y_2 = self._features_rest[:, [3, 4, 5, 6, 7]]
    # Y_2_rotated = torch.matmul(D_2, Y_2)
    # self._features_rest[:, [3, 4, 5, 6, 7]] = Y_2_rotated
    # print((sh - new_sh).abs().mean())
    return new_sh.reshape(B, 3, N).transpose(1, 2)


if __name__ == '__main__':
    with torch.no_grad():
        SP_GS_GUI().run()
