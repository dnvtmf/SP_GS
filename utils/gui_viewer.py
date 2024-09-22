from typing import Callable, Union
import math

import torch
import dearpygui.dearpygui as dpg
import numpy as np
from torch import Tensor
import cv2

from . import ops_3d


class ImageViewer:

    def __init__(self, image=None, size=(100, 100), channels=3, pad=0, tag='image', **kwargs) -> None:
        self.pad = pad
        if image is None:
            image = np.ones((size[1], size[0], channels), dtype=np.float32)
        assert image.ndim == 3 and image.shape[-1] in [3, 4]
        self.size = (image.shape[1], image.shape[0])
        self.channels = channels
        assert self.channels in [3, 4]
        self._data = (image.astype(np.float32) / 255) if image.dtype == np.uint8 else image.astype(np.float32)
        self._origin_data = None
        self._can_dynamic_change = False
        self.pad = pad
        self.tag = tag
        with dpg.texture_registry(show=False) as self._registry_id:
            # self.registry_id = registry_id
            self._texture_id = dpg.add_raw_texture(
                self.width,
                self.height,
                default_value=self._data,  # noqa
                format=dpg.mvFormat_Float_rgba if self.channels == 4 else dpg.mvFormat_Float_rgb,
                tag=tag
            )
        W, H = self.size
        self._win_id = dpg.add_window(
            width=W + 2 * self.pad, height=H + 2 * self.pad, no_title_bar=True, no_scrollbar=True, **kwargs
        )
        self._img_id = dpg.add_image(self.tag, width=W, height=H, parent=self._win_id)

        with dpg.theme() as container_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, self.pad, self.pad, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme(self._win_id, container_theme)

        self.resize_with_window()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_image: np.ndarray):
        assert new_image.shape == self._data.shape
        self._data[:] = (new_image / 255. if new_image.dtype == np.uint8 else new_image).astype(np.float32)

    @property
    def origin_data(self):
        if not self._can_dynamic_change:
            return self.data
        if self._origin_data is None:
            self._origin_data = self.data.copy()
        return self._origin_data

    @property
    def win_tag(self):
        return self._win_id

    @property
    def image_tag(self):
        return self._img_id

    @property
    def width(self) -> int:
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def resize_with_window(self):
        def resize_handler(sender):
            H, W = dpg.get_item_height(self._win_id), dpg.get_item_width(self._win_id)
            self.resize(W - 2 * self.pad, H - 2 * self.pad)

        with dpg.item_handler_registry() as hr_id:
            dpg.add_item_resize_handler(callback=resize_handler)
        dpg.bind_item_handler_registry(self._win_id, hr_id)

    def resize(self, W: int = None, H: int = None, channels: int = None):
        W = self.width if W is None else W
        H = self.height if H is None else H
        channels = self.channels if channels is None else channels
        if (W, H) == self.size and channels == self.channels:
            return False
        assert self.channels in [3, 4]
        new_image = np.ones((H, W, channels), dtype=np.float32)
        min_H, min_W, min_c = min(H, self.height), min(W, self.width), min(channels, self.channels)
        new_image[:min_H, :min_W, :min_c] = self.data[:min_H, :min_W, :min_c]
        self._data = new_image
        if self._origin_data is not None:
            new_image = np.ones_like(self.data)
            new_image[:min_H, :min_W, :min_c] = self._origin_data[:min_H, :min_W, :min_c]
            self._origin_data = new_image
        self.channels = channels
        self.size = W, H

        # console.log(f'resize "{self.tag}": W={W}, H={H}')
        dpg.delete_item(self.tag)
        dpg.remove_alias(self.tag)
        dpg.hide_item(self._img_id)  # can not delete old image due to segmentation fault (core dumped)

        self._texture_id = dpg.add_raw_texture(
            W,
            H,
            default_value=self.data,  # noqa
            format=dpg.mvFormat_Float_rgba if self.channels == 4 else dpg.mvFormat_Float_rgb,
            tag=self.tag,
            parent=self._registry_id
        )
        self._img_id = dpg.add_image(self._texture_id, parent=self._win_id)
        dpg.configure_item(self._win_id, width=W + 2 * self.pad, height=H + 2 * self.pad)
        return True

    def update(self, image: Union[np.ndarray, Tensor], resize=False):
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        elif image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=-1)
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose((1, 2, 0))
        if resize:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        else:
            self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image
        self._origin_data = None

    def get_mouse_pos(self):
        x, y = dpg.get_mouse_pos(local=False)
        wx, wy = dpg.get_item_pos(self._win_id)
        ix, iy = dpg.get_item_pos(self._img_id)
        return int(x - wx - ix), int(y - wy - iy)

    def enable_dynamic_change(self, hover_callback=None):
        self._can_dynamic_change = True
        if hover_callback is None:
            return
        with dpg.item_handler_registry() as handler:
            dpg.add_item_hover_handler(callback=hover_callback)
        dpg.bind_item_handler_registry(self._img_id, handler)


class ImagesGUI:

    def __init__(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title='ImagesGUI', width=800, height=600)
        with dpg.window(tag='Primary Window'):
            img = ImageViewer(pos=(300, 100), pad=5, no_move=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window('Primary Window', True)
        dpg.start_dearpygui()
        dpg.destroy_context()


class Viewer3D(ImageViewer):

    def __init__(self, renderer: Callable, size=(100, 100), pad=0, tag='3d', no_resize=True, no_move=True, **kwargs):
        super().__init__(size=size, pad=pad, tag=tag, no_resize=no_resize, no_move=no_move, **kwargs)

        self.renderer = renderer
        self.fovy = math.radians(60.)
        self.Tv2s = ops_3d.camera_intrinsics(size=size, fovy=self.fovy)
        self.Ts2v = ops_3d.camera_intrinsics(size=size, fovy=self.fovy, inv=True)

        self.up = torch.tensor([0, 1., 0.])
        self.eye = torch.tensor([0., 0., 2.0])
        self.at = torch.tensor([0., 0., 0.])
        #
        self._last_mouse_pos = None
        self._last_mouse_idx = None
        self.rate_rotate = self.fovy / self.height  # 旋转速度
        self.rate_translate = 1.  # 平移速度
        self.need_update = True

    def resize(self, W: int = None, H: int = None, channels: int = None):
        if super().resize(W, H, channels):
            self.need_update = True

    def callback_mouse_down(self, sender, app_data):
        # if dpg.is_item_hovered(self._img_id):
        #     self._last_mouse_pos = self.get_mouse_pos()
        #     self._last_mouse_idx = app_data[0]
        #     print(sender, app_data, self._last_mouse_pos)
        # else:
        #     self._last_mouse_pos = None
        #     self._last_mouse_idx = None
        pass

    def callback_mouse_release(self, sender, app_data):
        self._last_mouse_pos = None
        self._last_mouse_idx = None

    def callback_mouse_wheel(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        self.scale(app_data)

    def callback_mouse_drag(self, sender, app_data):
        if not dpg.is_item_hovered(self._img_id):
            return
        if app_data[0] == dpg.mvMouseButton_Left:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.rotate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        elif app_data[0] == dpg.mvMouseButton_Right:
            if self._last_mouse_pos is not None and self._last_mouse_idx == app_data[0]:
                now_pos = self.get_mouse_pos()
                self.translate(now_pos[0] - self._last_mouse_pos[0], now_pos[1] - self._last_mouse_pos[1])
        self._last_mouse_pos = self.get_mouse_pos()
        self._last_mouse_idx = app_data[0]

    def rotate(self, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return
        radiu = (self.eye - self.at).norm()
        dir_vec = ops_3d.normalize(self.eye - self.at)
        right_vec = ops_3d.normalize(torch.linalg.cross(self.up, dir_vec, dim=-1), dim=-1)
        theta = -dy * self.rate_rotate
        dir_vec = ops_3d.quaternion_xfm(dir_vec, ops_3d.quaternion_from_rotate(right_vec, right_vec.new_tensor(theta)))

        right_vec = ops_3d.normalize(torch.linalg.cross(self.up, dir_vec), dim=-1)
        up_vec = torch.linalg.cross(dir_vec, right_vec, dim=-1)
        theta = -dx * self.rate_rotate
        dir_vec = ops_3d.quaternion_xfm(dir_vec, ops_3d.quaternion_from_rotate(up_vec, up_vec.new_tensor(float(theta))))
        self.eye = self.at + ops_3d.normalize(dir_vec) * radiu
        self.up = up_vec
        self.need_update = True

    def translate(self, dx: float, dy: float):
        """在垂直于视线方向进行平移, 即在view space进行平移"""
        if dx == 0 and dy == 0:
            return
        Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
        p1 = ops_3d.xfm(ops_3d.xfm(self.at, Tw2v), self.Tv2s)

        p2 = p1.clone()
        p2[0] += dx * p1[2]
        p2[1] += dy * p1[2]
        Tv2w = ops_3d.look_at(self.eye, self.at, self.up, inv=True)
        p1 = ops_3d.xfm(ops_3d.xfm(p1, self.Ts2v), Tv2w)
        p2 = ops_3d.xfm(ops_3d.xfm(p2, self.Ts2v), Tv2w)
        delta = (p1 - p2)[:3] * self.rate_translate
        self.at += delta
        self.eye += delta
        self.need_update = True

    def scale(self, delta=0.0):
        self.eye = self.at + (self.eye - self.at) * 1.1 ** (-delta)
        self.need_update = True

    def update(self, image: Union[np.ndarray, Tensor] = None, resize=False):
        if image is None and not self.need_update:
            return
        self.need_update = False
        if image is None:
            Tw2v = ops_3d.look_at(self.eye, self.at, self.up)
            image = self.renderer(Tw2v, self.fovy, self.size)
        if isinstance(image, Tensor):
            image = image.detach().cpu().numpy()
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        image = image.astype(np.float32)
        if image.ndim == 4:
            image = image[0]
        if image.shape[-1] not in [3, 4]:
            assert image.shape[0] in [3, 4]
            image = image.transpose(1, 2, 0)
        if resize:
            image = cv2.resize(image, self.size)
        self.resize(image.shape[1], image.shape[0], image.shape[2])
        self.data = image

    def set_fovy(self, fovy=60.):
        self.fovy = math.radians(fovy)
        self.Tv2s = ops_3d.camera_intrinsics(size=self.size, fovy=self.fovy)
        self.Ts2v = ops_3d.camera_intrinsics(size=self.size, fovy=self.fovy, inv=True)
        self.need_update = True

    def set_pose(self, eye=None, at=None, up=None, Tw2v=None, Tv2w=None):
        if Tv2w is None and Tw2v is not None:
            Tv2w = Tw2v.inverse()
        if Tv2w is not None:
            Tv2w = Tv2w.view(-1, 4, 4)[0].to(self.eye.device)
            eye, at, up = ops_3d.look_at_get(Tv2w)
        if eye is not None:
            self.eye = eye
        if at is not None:
            self.at = at
        if up is not None:
            self.up = up
        self.need_update = True

    def set_need_update(self, need_update=True):
        self.need_update = need_update

    def build_gui_camera(self):
        with dpg.group(horizontal=True):
            dpg.add_text('fovy')
            dpg.add_slider_float(
                min_value=15.,
                max_value=180.,
                default_value=math.degrees(self.fovy),
                callback=lambda *args: self.set_fovy(dpg.get_value('set_fovy')),
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
                self.eye = self.eye.new_tensor([dpg.get_value(item) for item in ['eye_x', 'eye_y', 'eye_z']])
                self.at = self.at.new_tensor([dpg.get_value(item) for item in ['at_x', 'at_y', 'at_z']])
                self.need_update = True

            def to_camera_pos(campos, up):
                def callback():
                    r = (self.eye - self.at).norm()
                    eye = self.eye.new_tensor(campos)
                    self.eye = eye / eye.norm(keepdim=True) * r + self.at
                    self.up = self.up.new_tensor(up)
                    self.set_need_update()

                return callback

            with dpg.group(horizontal=True):
                dpg.add_button(label='change', callback=change_eye)
                dpg.add_button(label='+X', callback=to_camera_pos((1, 0, 0), (0, 1, 0)))
                dpg.add_button(label='-X', callback=to_camera_pos((-1, 0, 0), (0, 1, 0)))
                dpg.add_button(label='+Y', callback=to_camera_pos((0, 1, 0), (0, 0, 1)))
                dpg.add_button(label='-Y', callback=to_camera_pos((0, -1, 0), (0, 0, 1)))
                dpg.add_button(label='+Z', callback=to_camera_pos((0, 0, 1), (0, 1, 0)))
                dpg.add_button(label='-Z', callback=to_camera_pos((0, 0, -1), (0, 1, 0)))

    def update_gui_camera(self):
        if self.need_update:
            dpg.set_value('eye_x', self.eye[0].item())
            dpg.set_value('eye_y', self.eye[1].item())
            dpg.set_value('eye_z', self.eye[2].item())
            dpg.set_value('at_x', self.at[0].item())
            dpg.set_value('at_y', self.at[1].item())
            dpg.set_value('at_z', self.at[2].item())


def simple_3d_viewer(rendering, size=(400, 400)):
    dpg.create_context()
    dpg.create_viewport(title='Custom Title')
    with dpg.window(tag='Primary Window'):
        img = Viewer3D(rendering, size=size, no_resize=False, no_move=True)
        with dpg.window(tag='control', width=256):
            dpg.add_text(tag='fps')
            img.build_gui_camera()

    with dpg.handler_registry():
        dpg.add_mouse_drag_handler(callback=img.callback_mouse_drag)
        dpg.add_mouse_wheel_handler(callback=img.callback_mouse_wheel)
        dpg.add_mouse_release_handler(callback=img.callback_mouse_release)

    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window('Primary Window', True)
    # dpg.start_dearpygui()
    last_size = None
    while dpg.is_dearpygui_running():
        dpg.render_dearpygui_frame()
        img.update_gui_camera()
        img.update()
        now_size = dpg.get_item_width(img._win_id), dpg.get_item_height(img._win_id)
        if last_size != now_size:
            dpg.configure_item('control', pos=(dpg.get_item_width(img._win_id), 0))
            dpg.set_viewport_width(dpg.get_item_width(img._win_id) + dpg.get_item_width('control'))
            dpg.set_viewport_height(dpg.get_item_height(img._win_id))
            last_size = now_size
        dpg.set_value('fps', f"FPS: {dpg.get_frame_rate()}")
    dpg.destroy_context()
