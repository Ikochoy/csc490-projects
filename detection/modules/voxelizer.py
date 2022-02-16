from dataclasses import dataclass
from time import sleep
from typing import List, Tuple

import torch

from detection.types import Detections


@dataclass
class VoxelizerConfig:
    """Configuration class for the voxelizer.

    Attributes:
        x_range: Image range along the x axis in metres ("forward")
        y_range: Image range along the y axis in metres ("sideways")
        z_range: Image range along the z axis in metres ("up")
        step: Voxelization step in metres (i.e. pixel resolution)
    """

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
    step: float

    @property
    def bev_size(self) -> Tuple[int, int, int]:
        """Return (depth, height, width) of the voxel grid.

        depth is the size along the z-axis ("up"), height is the size along the y-axis ("sideways"),
        and width is the size along the x-axis ("forward").
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        z_min, z_max = self.z_range
        return (
            int((z_max - z_min) / self.step),
            int((y_max - y_min) / self.step),
            int((x_max - x_min) / self.step),
        )


class Voxelizer(torch.nn.Module):
    """Voxelizer for converting Lidar point cloud to image"""

    def __init__(self, config: VoxelizerConfig) -> None:
        super().__init__()
        self._step = config.step
        self._x_min, self._x_max = config.x_range
        self._y_min, self._y_max = config.y_range
        self._z_min, self._z_max = config.z_range
        self._depth, self._height, self._width = config.bev_size
        print("CONFIG:", self._x_min)

    def forward(self, pointclouds: List[torch.Tensor]) -> torch.Tensor:
        """Voxelize a batch of lidar pointclouds into a BEV occupacy image.

        This method produces a [batch_size x D x H x W] tensor, where D is the
        size of the voxel grid along the z-axis, H is that along the y-axis, and
        W is that along the x-axis. Each cell in a [D x H x W] voxel grid denotes
        whether a LiDAR point occupies the cell (1) or not (0).

        A LiDAR point (x, y, z) occupies a cell (i, j, k) if and only if:
            floor[(z - z_min) / step] = i
            floor[(y_max - y) / step] = j
            floor[(x - x_min) / step] = k

        The z coordinate of all LiDAR points are clamped between [z_min, z_max].
        The x and y coordinates of all LiDAR points are not. Therefore, if a point
        with coordinates (x, y, z) lies outside the bounds of [x_min, x_max] and
        [y_min, y_max], it will not be represented in the voxel grid.

        Args:
            pointclouds: Batch of [N x 3] lidar pointclouds. Each row is (x, y, z).

        Returns:
            BEV occupacy image as a [batch_size x D x H x W] tensor.
        """
        ## pointcloud = point; batch of pointclouds = one frame captured
        # TODO: Replace this stub code.
        initial_tensor = torch.zeros(
            (len(pointclouds), self._depth, self._height, self._width),
            dtype=torch.bool,  ## or float??
            device=pointclouds[0].device,
        )
        for w, cloud in enumerate(pointclouds):
            #line 89- 99 tested
            idx_within_range = torch.bitwise_and(
                                torch.bitwise_and(
                                    torch.bitwise_and(cloud[:, 0] > self._x_min, cloud[:, 0] < self._x_max),
                                    torch.bitwise_and(cloud[:, 1] > self._y_min, cloud[:, 1] < self._y_max),
                                ), 
                                torch.bitwise_and(cloud[:, 2] > self._z_min, cloud[:, 2] < self._z_max)  
                            )
            cloud = cloud[idx_within_range]
            cloud[:, 0].add_(-self._x_min)
            torch.neg_(cloud[:, 1])
            cloud[:, 1].add_(self._y_max)
            cloud[:, 2].add_(-self._z_min)
            # new cloud is currently in x, y, z order i.e: k, j, i order
            new_cloud = torch.div(cloud, self._step, rounding_mode='floor')
            new_cloud = torch.fliplr(new_cloud)
            # now new_cloud is in i, j, k order after left right flip


            # print(new_cloud[:, 2].shape, new_cloud[:, 1].shape, new_cloud[:, 0].shape)
            # print(new_cloud[:, 2])
            # sleep(5)
            # new_cloud = torch.cat(torch.unsqueeze((new_cloud[:, 2], 0), torch.unsqueeze(new_cloud[:, 1], 0), torch.unsqueeze(new_cloud[:, 0], 0)), 1)  ##
            
            # x -> k, y -> j, z -> i
            # padding_column = torch.unsqueeze(torch.tensor([w] * new_cloud.shape[0]), 1)
            # padded_w = torch.cat([padding_column, new_cloud], 1)  ##
            # indices = padded_w.long()
            indices = new_cloud.long()
            
            # ones = torch.ones(indices.shape[0],
            #                   dtype=torch.bool,
            #                   device=pointclouds[0].device)
            ones = torch.tensor(1.).bool()
            # print(initial_tensor.type(), ones.type())
            # initial_tensor.index_put_(tuple(indices.t()), ones)
            # initial_tensor[w].index_put_(tuple(indices.t()), ones)
            initial_tensor[w].index_put_(tuple(indices.t()), ones)

            # for row_point in cloud:
            #     x, y, z = row_point[0].item(), row_point[1].item(), row_point[2].item()
            #     # if x < self._x_max and x > self._x_min and y > self._y_min
            #     i, j, k = int((z - self._z_min) // self._step), int((self._y_max - y) // self._step), int((x - self._x_min) // self._step)
            #     # i, j, k = torch.floor_divide(torch.tensor(i,j,k), self._step)
            #     initial_tensor[w, i, j, k] = 1
        return initial_tensor


    def project_detections(self, detections: Detections) -> Detections:
        """Project detections to voxelized frame and filter out-of-bounds ones.

        Args:
            detections: 2D bounding box detections, in vehicle coordinates.

        Returns:
            2D bounding box detections, in voxel grid coordinates.
        """
        # Remove out of bounds detections
        out_of_bounds = torch.any(
            torch.stack(
                [
                    detections.centroids[:, 0] < self._x_min,
                    detections.centroids[:, 0] > self._x_max,
                    detections.centroids[:, 1] < self._y_min,
                    detections.centroids[:, 1] > self._y_max,
                ]
            ),
            dim=0,
        )
        mask = ~out_of_bounds

        # Transform vehicle coordinates to image
        x, y = detections.centroids[mask, :2].T
        ix = (x - self._x_min) / self._step
        iy = (self._y_max - y) / self._step
        centroids = torch.stack([ix, iy], dim=1)

        # Transform vehicle-frame relative lengths to image lengths (pixels)
        boxes = detections.boxes[mask, :2] / self._step

        # Yaws are unchanged
        yaws = detections.yaws[mask]

        # Scores are unchanged
        scores = detections.scores[mask] if detections.scores is not None else None

        return Detections(centroids, yaws, boxes, scores)
