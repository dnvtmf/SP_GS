import torch
from torch import Tensor


def furthest_point_sampling(points: Tensor, n_samples: int) -> Tensor:
    """

    Args:
        points: [N, 3], tensor containing the whole point cloud
        n_samples: samples you want in the sampled point cloud typically << N

    Returns:
        Tensor
    """
    N, _ = points.shape
    sampled = []
    # first
    selected = torch.randint(0, N, size=(1,)).item()
    sampled.append(selected)
    dist = (points - points[selected]).norm(dim=-1)
    dist[selected] = -1
    for i in range(1, n_samples):
        selected = dist.argmax().item()
        sampled.append(selected)
        dist_i = (points - points[selected]).norm(dim=-1)
        dist_i[selected] = -1
        dist = torch.minimum(dist, dist_i)
    return torch.tensor(sampled, device=points.device)
