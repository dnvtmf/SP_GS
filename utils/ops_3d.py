""" OPENCV coordinate transformation 三维坐标变换

世界空间World space和观察空间view space为右手系
(COLMAP坐标系) RDF,  +x 朝右, +y 朝下,  +z 朝里
COLMAP/OpenCV:
     z
   ↗
  .-------> x
  |
  |
  ↓
  y
裁剪空间Clip space为左手系: +x指向右手边, +y 指向上方, +z指向屏幕内; z 的坐标值越小，距离观察者越近
y [-1, 1]
↑
|   z [-1, 1]
| ↗
.------> x [-1, 1]
屏幕坐标系： X 轴向右为正，Y 轴向下为正，坐标原点位于窗口的左上角 (左手系: z轴向屏幕内，表示深度)
    z
  ↗
.------> x
|
|
↓
y
坐标变换矩阵: T{s}2{d} Transform from {s} space to {d} space
{s} 和 {d} 包括世界World坐标系、观察View坐标系、裁剪Clip坐标系、屏幕Screen坐标

Tv2s即相机内参，Tw2v即相机外参
"""

from typing import Tuple, Union, Optional, Sequence
import math
import torch
import numpy as np
from torch import Tensor
import torch.nn.functional as F

TensorType = Union[float, int, np.ndarray, Tensor]


def to_tensor(x, dtype=None, device=None, **kwargs) -> Optional[Tensor]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        y = x
    elif isinstance(x, np.ndarray):
        y = torch.from_numpy(x)
    # elif isinstance(x, (list, tuple)):
    #     x = np.ndarray(x)
    #     x = torch.from_numpy(x)
    else:
        y = torch.tensor(x, dtype=dtype, device=device)
    if dtype is not None:
        y = y.type(dtype)
    if device is not None:
        y = y.to(device=device, **kwargs)
    return y


def as_np_image(img: Union[Tensor, np.ndarray]) -> np.ndarray:
    """convert the [0, 1.]  tensor to [0, 255] uint8 image """
    if isinstance(img, Tensor):
        return img.detach().clamp(0, 1).mul_(255.).cpu().numpy().astype(np.uint8)
    elif img.dtype != np.uint8:
        return np.clip(img * 255, 0, 255).astype(np.uint8)
    else:
        return img


def normalize(x: Tensor, dim=-1, eps=1e-20):
    return F.normalize(x, dim=dim, eps=eps)
    # return x / torch.sqrt(torch.clamp(torch.sum(x * x, dim, keepdim=True), min=eps))


def dot(x: Tensor, y: Tensor, keepdim=True, dim=-1) -> Tensor:
    """Computes the dot product of two tensors in the given dimension dim"""
    return torch.sum(x * y, dim, keepdim=keepdim)


def fovx_to_fovy(fovx, aspect=1.) -> Union[np.ndarray, Tensor]:
    if isinstance(fovx, Tensor):
        return torch.arctan(torch.tan(fovx * 0.5) / aspect) * 2.0
    else:
        return np.arctan(np.tan(fovx * 0.5) / aspect) * 2.0


def focal_to_fov(focal: Union[float, Tensor, np.ndarray], *size: Union[float, Sequence[float]]):
    """focal length of fov"""
    if len(size) == 1:
        size = size[0]
    if isinstance(size, Sequence):
        if isinstance(focal, Tensor):
            return torch.stack([2 * torch.arctan(0.5 * s / focal) for s in size], dim=-1)
        else:
            return np.stack([2 * np.arctan(0.5 * s / focal) for s in size], axis=-1)
    else:
        t = 0.5 * size / focal
        return 2 * (torch.arctan(t) if isinstance(t, Tensor) else np.arctan(t))


def xfm(points: Tensor, matrix: Tensor, homo=False) -> Tensor:
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [..., num_vertices, 3] or [..., num_vertices, 4]
        matrix: A 4x4 transform matrix with shape [..., 4, 4] or [4, 4]
        homo: convert output to homogeneous
    Returns:
        Tensor: Transformed points in homogeneous 4D with shape [..., num_vertices, 3/4]
    '''
    is_points = True
    dim = points.shape[-1]
    if dim + 1 == matrix.shape[-1]:
        points = torch.constant_pad_nd(points, (0, 1), 1.0 if is_points else 0.0)
    else:
        homo = False
    if is_points:
        out = torch.matmul(points, torch.transpose(matrix, -1, -2))
    else:
        out = torch.matmul(points, torch.transpose(matrix, -1, -2))
    if not homo:
        out = out[..., :dim]
    return out


def fov_to_focal(fov: Union[float, Tensor, np.ndarray], size: Union[float, Tensor, np.ndarray]):
    """FoV to focal length"""
    return size / (2 * (torch.tan(fov * 0.5) if isinstance(fov, Tensor) else np.tan(fov * 0.5)))


## 世界坐标系相关
def coord_spherical_to(radius: TensorType, thetas: TensorType, phis: TensorType) -> Tensor:
    """ 球坐标系 转 笛卡尔坐标系

    Args:
        radius: 径向半径 radial distance, 原点O到点P的距离 [0, infity]
        thetas: 极角 polar angle, -y轴与连线OP的夹角 [0, pi]
        phis: 方位角 azimuth angle, 正x轴与连线OP在xz平面的投影的夹角, [0, 2 * pi], 顺时针, +z轴 -0.5pi
    Returns:
        Tensor: 点P的笛卡尔坐标系, shape: [..., 3]
    """
    radius = to_tensor(radius, dtype=torch.float32)
    thetas = to_tensor(thetas, dtype=torch.float32)
    phis = to_tensor(phis, dtype=torch.float32)
    # yapf: disable
    return torch.stack([
            radius * torch.sin(thetas) * torch.cos(phis),
            -radius * torch.cos(thetas),
            -radius * torch.sin(thetas) * torch.sin(phis),
        ], dim=-1)
    # yapf: enable


def coord_to_spherical(points: Tensor):
    """ 笛卡尔坐标系(OpenGL) 转 球坐标系

    Args:
        points (Tensor): 点P的笛卡尔坐标系, shape: [..., 3]

    Returns:
        Tuple[Tensor, Tensor, Tensor]: 点P的球坐标系: 径向半径、极角和方位角
    """
    raduis = points.norm(p=2, dim=-1)  # type: Tensor
    thetas = torch.arccos(-points[..., 1] / raduis)
    phis = torch.arctan2(-points[..., 2], points[..., 0])
    return raduis, thetas, phis


## 相机坐标系相关
def look_at(eye: Tensor, at: Tensor = None, up: Tensor = None, inv=False) -> Tensor:
    if at is None:
        dir_vec = torch.zeros_like(eye)
        dir_vec[..., 3] = 1.
    else:
        dir_vec = normalize(eye - at)

    if up is None:
        up = torch.zeros_like(dir_vec)
        # if dir is parallel with y-axis, up dir is z axis, otherwise is y-axis
        y_axis = dir_vec.new_tensor([0, -1., 0]).expand_as(dir_vec)
        y_axis = torch.cross(dir_vec, y_axis, dim=-1).norm(dim=-1, keepdim=True) < 1e-6
        up = torch.scatter_add(up, -1, y_axis + 1, 1 - y_axis.to(up.dtype) * 2)
    shape = eye.shape
    right_vec = normalize(torch.cross(up, dir_vec, dim=-1))  # 相机空间x轴方向
    up_vec = torch.cross(right_vec, dir_vec, dim=-1)  # 相机空间y轴方向
    if inv:
        Tv2w = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        Tv2w[..., :3, 0] = right_vec
        Tv2w[..., :3, 1] = up_vec
        Tv2w[..., :3, 2] = dir_vec
        Tv2w[..., :3, 3] = eye
        return Tv2w
    else:
        R = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        T = torch.eye(4, device=eye.device, dtype=eye.dtype).expand(*shape[:-1], -1, -1).contiguous()
        R[..., 0, :3] = right_vec
        R[..., 1, :3] = up_vec
        R[..., 2, :3] = dir_vec
        T[..., :3, 3] = eye
        world2view = R @ T
        return world2view


def look_at_get(Tv2w: Tensor):
    eye = Tv2w[..., :3, 3]
    right_vec = Tv2w[..., :3, 0]
    up_vec = Tv2w[..., :3, 1]
    dir_vec = Tv2w[..., :3, 2]
    at = eye - dir_vec
    return eye, at, torch.cross(dir_vec, right_vec, dim=-1)


def camera_intrinsics(focal=None, cx_cy=None, size=None, fovy=np.pi, inv=False, **kwargs) -> Tensor:
    """生成相机内参K/Tv2s, 请注意坐标系
    .---> u
    |
    ↓
    v
    """
    W, H = size
    if focal is None:
        focal = fov_to_focal(fovy, H)
    if cx_cy is None:
        cx, cy = 0.5 * W, 0.5 * H
    else:
        cx, cy = cx_cy
    shape = [x.shape for x in [focal, cx, cy] if isinstance(x, Tensor)]
    if len(shape) > 0:
        shape = list(torch.broadcast_shapes(*shape))
    if inv:  # Ts2v
        fr = 1. / focal
        Ts2v = torch.zeros(shape + [3, 3], **kwargs)
        Ts2v[..., 0, 0] = fr
        Ts2v[..., 0, 2] = cx * fr
        Ts2v[..., 1, 1] = fr
        Ts2v[..., 1, 2] = cy * fr
        Ts2v[..., 2, 2] = 1
        return Ts2v
    else:
        K = torch.zeros(shape + [3, 3], **kwargs)  # Tv2s
        K[..., 0, 0] = focal
        K[..., 0, 2] = cx
        K[..., 1, 1] = focal
        K[..., 1, 2] = cy
        K[..., 2, 2] = 1
        return K


def perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None, size=None):
    """透视投影矩阵

    Args:
        fovy: 弧度. Defaults to 0.7854.
        aspect: 长宽比W/H. Defaults to 1.0.
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.
        size: (W, H)

    Returns:
        Tensor: 透视投影矩阵
    """
    shape = []
    if size is not None:
        aspect = size[0] / size[1]
    for x in [fovy, aspect, n, f]:
        if isinstance(x, Tensor):
            shape = x.shape
    Tv2c = torch.zeros(*shape, 4, 4, dtype=torch.float, device=device)
    y = np.tan(fovy * 0.5)
    Tv2c[..., 0, 0] = 1. / (y * aspect)
    Tv2c[..., 1, 1] = -1. / y
    Tv2c[..., 2, 2] = -(f + n) / (f - n)
    Tv2c[..., 2, 3] = -(2 * f * n) / (f - n)
    Tv2c[..., 3, 2] = -1
    return Tv2c


def perspective_v2(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None, size=None):
    """透视投影矩阵

    Args:
        fovy: 弧度. Defaults to 0.7854.
        aspect: 长宽比W/H. Defaults to 1.0.
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.
        size: (W, H)

    Returns:
        Tensor: 透视投影矩阵
    """
    shape = []
    if size is not None:
        aspect = size[0] / size[1]
    for x in [fovy, aspect, n, f]:
        if isinstance(x, Tensor):
            shape = x.shape
    Tv2c = torch.zeros(*shape, 4, 4, dtype=torch.float, device=device)
    y = np.tan(fovy * 0.5)
    x = y * aspect
    top = y * n
    bottom = -top
    right = x * n
    left = -right
    z_sign = 1.0

    Tv2c[..., 0, 0] = 2.0 * n / (right - left)
    Tv2c[..., 1, 1] = 2.0 * n / (top - bottom)
    Tv2c[..., 0, 2] = (right + left) / (right - left)
    Tv2c[..., 1, 2] = (top + bottom) / (top - bottom)
    Tv2c[..., 3, 2] = z_sign
    Tv2c[..., 2, 2] = z_sign * f / (f - n)
    Tv2c[..., 2, 3] = -(f * n) / (f - n)
    return Tv2c


# @try_use_C_extension
def ortho(l=-1., r=1.0, b=-1., t=1.0, n=0.1, f=1000.0, device=None):
    """正交投影矩阵

    Args:
        # size: 长度. Defaults to 1.0.
        # aspect: 长宽比W/H. Defaults to 1.0.
        l: left plane
        r: right plane
        b: bottom place
        t: top plane
        n: near. Defaults to 0.1.
        f: far. Defaults to 1000.0.
        device: Defaults to None.

    Returns:
        Tensor: 正交投影矩阵
    """
    raise NotImplementedError
    # yapf: disable
    return torch.tensor([
        [2/(r-l), 0,       0,       (l+r)/(l-r)],
        [0,       2/(t-b), 0,       (b+t)/(b-t)],
        [0,       0,       2/(n-f), (n+f)/(n-f)],
        [0,       0,       0,       1],
    ], dtype=torch.float32, device=device)
    # return torch.tensor([
    #     [1 / (size * aspect), 0, 0, 0],
    #     [0, 1 / size, 0, 0],
    #     [0, 0, -(f + n) / (f - n), -(f + n) / (f - n)],
    #     [0, 0, 0, 0],
    # ], dtype=torch.float32, device=device)
    # yapf: enable


def convert_coord_system(T: Tensor, src='opengl', dst='opengl', inverse=False) -> Tensor:
    """ convert coordiante system from <source> to <goal>: p_dst = M @ p_src
    Args:
        T: transformation matrix with shape [..., 4, 4], can be Tw2v or Tv2w
        src: the source coordinate system, must be blender, colmap, opencv, llff, PyTorch3d, opengl
        dst: the destination coordinate system
        inverse: inverse apply (used when T is Tv2w)
    Returns:
        Tensor: converted transformation matrix
    """
    if src == dst:
        return T
    if inverse:
        src, dst = dst, src
    mapping = {
        'opengl': 'opengl',
        'blender': 'blender',
        'colmap': 'opencv',
        'opencv': 'opencv',
        'llff': 'llff',
        'pytorch3d': 'pytorch3d',
    }
    src = mapping[src.lower()]
    dst = mapping[dst.lower()]
    if src == 'opengl':
        M = T.new_tensor({
            'blender': [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]],
            'opencv': [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
            'llff': [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'blender':
        M = T.new_tensor({
            'opengl': [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.]],
            'opencv': [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.]],
            'llff': [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'opencv':
        M = T.new_tensor({
            'opengl': [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
            'blender': [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.]],
            'llff': [[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'llff':
        M = T.new_tensor({
            'opengl': [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]],
            'blender': [[0, 0, -1, 0], [0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'opencv': [[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1.]],
            'pytorch3d': [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
        }[dst])
    elif src == 'pytorch3d':
        M = T.new_tensor({
            'opengl': [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1.]],
            'blender': [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.]],
            'opencv': [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1.]],
            'llff': [[0, -1, 0, 0], [-1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1.]],
        }[dst])
    else:
        raise NotImplementedError(f"src={src}, dst={dst}")
    return T @ M if inverse else M @ T


def translate(*xyz: Union[float, Tensor], dtype=None, device=None):
    if len(xyz) == 1:
        if isinstance(xyz[0], Tensor):
            t = xyz[0]
            assert t.shape[-1] == 3
            dtype = t.dtype if dtype is None else dtype
            device = t.device if device is None else device
            T = torch.eye(4, dtype=dtype, device=device).expand(list(t.shape[:-1]) + [4, 4]).contiguous()
            T[..., :3, 3] = t
            return T
        else:
            assert isinstance(xyz[0], (list, tuple))
            x, y, z = xyz[0]
    else:
        assert len(xyz) == 3
        x, y, z = xyz
    shape = []
    for t in [x, y, z]:
        if isinstance(t, Tensor):
            dtype = t.dtype if dtype is None else dtype
            device = t.device if device is None else device
            shape.append(t.shape)
    if not shape:
        return torch.tensor([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1.]], dtype=dtype, device=device)
    else:
        shape = torch.broadcast_shapes(*shape)
        T = torch.eye(4, dtype=dtype, device=device).expand(list(shape) + [4, 4]).contiguous()
        T[..., 0, 3] = x
        T[..., 1, 3] = y
        T[..., 2, 3] = z
        return T


def _rotate(angle: Union[float, Tensor], device=None, a=0, b=1) -> Tensor:
    if isinstance(angle, Tensor):
        s, c = torch.sin(angle), torch.cos(angle)
        T = torch.eye(4, dtype=s.dtype, device=s.device if device is None else device)
        T = T.expand(list(s.shape) + [4, 4]).contiguous()
        T[..., a, a] = c
        T[..., a, b] = -s
        T[..., b, a] = s
        T[..., b, b] = c
    else:
        s, c = math.sin(angle), math.cos(angle)
        T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        T[a][a] = c
        T[b][b] = c
        T[a][b] = -s
        T[b][a] = s
        T = torch.tensor(T, device=device)
    return T


def rotate_x(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 1, 2)


def rotate_y(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 0, 2)


def rotate_z(angle: Union[float, Tensor], device=None):
    return _rotate(angle, device, 0, 1)


def rotate(x: float = None, y: float = None, z: float = None, device=None):
    R = torch.eye(4, device=device)
    if x is not None:
        R = R @ rotate_x(x, device)
    if y is not None:
        R = R @ rotate_y(y, device)
    if z is not None:
        R = R @ rotate_z(z, device)
    return R


def scale(s: float, device=None):
    return torch.tensor([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]], dtype=torch.float32, device=device)


def quaternion_from_rotate(u: Tensor, theta: Tensor):
    """ 从旋转轴u和旋转角θ构造四元数

    Args:
        u: 旋转轴, 单位向量, shape: [..., 3]
        theta: 旋转角, 弧度; shape [....]
    Returns:
        四元数 [..., 4]
    """
    theta = theta[..., None] * 0.5
    return torch.cat([theta.sin() * u, theta.cos()], dim=-1)


def direction_vector_to_quaternion(before: Tensor, after: Tensor) -> Tensor:
    before = normalize(before)
    after = normalize(after)
    theta = torch.acos(dot(before, after))
    axis = torch.linalg.cross(before, after)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    return axis_angle_to_quaternion(axis, theta[..., 0])


def quaternion_mul(q: Tensor, s: Union[float, Tensor]):
    if isinstance(s, float):
        return q * s
    elif isinstance(s, Tensor):
        if q.ndim > s.ndim:
            return q * s[..., None]
        else:
            assert s.shape[-1] == 4
            # yapf: disable
            # b, c, d, a = q.unbind(-1)
            # f, g, h, e = q.unbind(-1)
            # return torch.stack([
            #     a * e - b * f - c * g - d * h,
            #     b * e + a * f - d * g + c * h,
            #     c * e + d * f + a * g - b * h,
            #     d * e - c * f + b * g + a * h,
            # ], dim=-1)
            # #(Graßmann Product)
            qxyz, qw = q[..., :3], q[..., -1:]
            sxyz, sw = s[..., :3], s[..., -1:]
            return torch.cat([
                sw * qxyz + qw * sxyz + torch.linalg.cross(qxyz, sxyz),
                qw * sw - torch.linalg.vecdot(qxyz, sxyz)[..., None]
            ], dim=-1)
            # yapf: enable
    else:
        raise ValueError()


def uaternion_conj(q: Tensor):
    """共轭 conjugate"""
    return torch.cat([-q[..., :3], q[..., -1:]], dim=-1)


def quaternion_xfm(points: Tensor, q: Tensor):
    """使用四元数旋转点"""
    points = torch.cat([points, torch.zeros_like(points[..., :1])], dim=-1)
    return quaternion_mul(quaternion_mul(q, points), uaternion_conj(q))[..., :3]


def axis_angle_to_quaternion(u: Tensor, theta: Tensor = None):
    if theta is None:
        theta = u.norm(dim=-1, keepdim=False)
        u = u / theta[..., None]
    q = quaternion_from_rotate(u, theta)
    return q
