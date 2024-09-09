import math
import random
import os
from dataclasses import dataclass, field
import bisect

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

import threestudio
import pdb
import curses
import math

import sys

from gsgen.utils.camera import CameraInfo

@dataclass
class GSEditRandomCameraDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 64
    width: Any = 64
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 90)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (1, 1.5)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.1
    center_perturb: float = 0.2
    up_perturb: float = 0.02
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 1.5
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    shape_init: bool = False
    fix_fov: bool = False
    init_steps: int = 1000
    enable_edit: bool = False # edit
    edit_bs: int = 1
    edit_n_view: int = 1
    edit_width: int = 128
    edit_height: int = 128
    edit_camera_distance_range: Tuple[float, float]= (0.8, 1.0) # relative
    edit_fovy_range: Tuple[float, float] = (15, 60)
    edit_elevation_range: Tuple[float, float] = (0, 30)
    edit_azimuth_range: Tuple[float, float] = (-180, 180)
    edit_relative_radius: bool = False
    start_edit_step: int = 600
    rotate_init: int = 0
    near_far: Tuple[float, float] = (0.1, 1000.0)
    fix_fov_value: float = 0.7
    disable_relative_radius: bool = False
    fix_view: bool = False

class GSEditRandomCameraIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg: GSEditRandomCameraDataModuleConfig = cfg
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.edit_heights: List[int] = (
            [self.cfg.edit_height] if isinstance(self.cfg.edit_height, int) else self.cfg.edit_height
        )
        self.edit_widths: List[int] = (
            [self.cfg.edit_width] if isinstance(self.cfg.edit_width, int) else self.cfg.edit_width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
            len(self.heights) == 1
            and len(self.widths) == 1
            and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.directions_unit_focals_edit = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.edit_heights, self.edit_widths)
        ]
        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.directions_unit_focal_edit = self.directions_unit_focals_edit[0]
        self.elevation_range = self.cfg.elevation_range
        self.azimuth_range = self.cfg.azimuth_range
        self.camera_distance_range = self.cfg.camera_distance_range
        self.fovy_range = self.cfg.fovy_range

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        threestudio.debug(
            f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        r = min(1.0, global_step / (self.cfg.progressive_until + 1))
        self.elevation_range = [
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[0],
            (1 - r) * self.cfg.eval_elevation_deg + r * self.cfg.elevation_range[1],
        ]
        self.azimuth_range = [
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[0],
            (1 - r) * 0.0 + r * self.cfg.azimuth_range[1],
        ]
        # self.camera_distance_range = [
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[0],
        #     (1 - r) * self.cfg.eval_camera_distance
        #     + r * self.cfg.camera_distance_range[1],
        # ]
        # self.fovy_range = [
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[0],
        #     (1 - r) * self.cfg.eval_fovy_deg + r * self.cfg.fovy_range[1],
        # ]

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(self.batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            )
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(self.batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            )
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        if self.cfg.batch_uniform_azimuth:
            # ensures sampled azimuth angles in a batch cover the whole range
            azimuth_deg = (
                torch.rand(self.batch_size) + torch.arange(self.batch_size)
            ) / self.batch_size * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
        else:
            # simple random sampling
            azimuth_deg = (
                torch.rand(self.batch_size)
                * (self.azimuth_range[1] - self.azimuth_range[0])
                + self.azimuth_range[0]
            )
        azimuth = azimuth_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        )

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(self.batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        )
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.center_perturb
        )
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(self.batch_size, 3) * self.cfg.up_perturb
        )
        up = up + up_perturb

        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(self.batch_size) * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        )
        fovy = fovy_deg * math.pi / 180

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(self.batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        )

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(self.batch_size, 3) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(self.batch_size) * math.pi - 2 * math.pi
            )  # [-pi, pi]
            light_elevation = (
                torch.rand(self.batch_size) * math.pi / 3 + math.pi / 6
            )  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
        }
    


@dataclass
class GSEditRandomMultiviewCameraDataModuleConfig(GSEditRandomCameraDataModuleConfig):
    relative_radius: bool =  True
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)


class GSEditRandomMultiviewCameraIterableDataset(GSEditRandomCameraIterableDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_range = self.cfg.zoom_range
        self.shape_init = self.cfg.shape_init
        self.fov_value = self.cfg.fix_fov_value # 0.7
        self.init_cnt = 0
        self.cnt = 0
        self.edit_mode = True
        self.edit_center = [0., 0., 0.]
        self.recenter_edit_origin = [0., 0., 0.]
        self.edit_prompt = None
        # simutaneously supervise multiple views
        self.edit_prompt_list = []
        self.edit_center_list = []
        self.recenter_edit_origin_list = []
        self.edit_elevation_range_list = []
        self.edit_azimuth_range_list = []
        self.edit_camera_distance_range_list = []
        self.save_edit_flag = True
        self.fix_view = self.cfg.fix_view
    
    def add_edit_part(self):
        self.edit_prompt_list.append(self.edit_prompt)
        self.edit_center_list.append(self.edit_center)
        self.recenter_edit_origin_list.append(self.recenter_edit_origin)
        self.edit_elevation_range_list.append(self.cfg.edit_elevation_range)
        self.edit_azimuth_range_list.append(self.cfg.edit_azimuth_range)
        self.edit_camera_distance_range_list.append(self.cfg.edit_camera_distance_range)
        self.recenter_edit_origin = [0., 0., 0]
        self.edit_center = [0., 0., 0]
        self.cfg.edit_camera_distance_range = [3.5, 3.5]
        self.cfg.edit_elevation_range = [30, 40]
        self.cfg.edit_azimuth_range = [0, 0]
        self.edit_prompt = None
    
    def save_edit(self):
        import json
        data = {}
        for i, prompt in enumerate(self.edit_prompt_list):
            data[prompt] = {'edit_center': list(self.edit_center_list[i]),
                            'recenter_origin': list(self.recenter_edit_origin_list[i]),
                            'edit_ele': list(self.edit_elevation_range_list[i]),
                            'edit_azi': list(self.edit_azimuth_range_list[i]),
                            'edit_dist': list(self.edit_camera_distance_range_list[i]),
                            }
        with open('debug_data/edit_data.json', 'w') as f:
            json.dump(data, f)
        print("edit data has been saved to debug_data/edit_data.json")
    
    def random_edit(self):
        total_parts = len(self.edit_prompt_list)
        if total_parts == 0:
            return
        idx = random.randint(0, total_parts - 1)
        self.edit_prompt = self.edit_prompt_list[idx]
        self.edit_center = self.edit_center_list[idx]
        self.recenter_edit_origin = self.recenter_edit_origin_list[idx]
        self.cfg.edit_elevation_range = self.edit_elevation_range_list[idx]
        self.cfg.edit_azimuth_range = self.edit_azimuth_range_list[idx]
        self.cfg.edit_camera_distance_range = self.edit_camera_distance_range_list[idx]
    
    def _interactive_mode(self, stdscr):
        curses.noecho()
        curses.cbreak()
        stdscr.keypad(1)

        stdscr.addstr(0, 10, f"Press 'a d, w x, z e, + -, arrows, or q'..., current params: (1) center: {self.edit_center}, (2) elevation: {self.cfg.edit_elevation_range}, azimuth: {self.cfg.edit_azimuth_range}, (3) camera distance: {self.cfg.edit_camera_distance_range}, (4) edit prompt: {self.edit_prompt}, (5) hw: {self.cfg.edit_height}")
        stdscr.refresh()

        while True:
            key = stdscr.getch()
            # center
            if key == ord('a'):
                assert sum(self.recenter_edit_origin) == 0, "can not change center after recentered!"
                self.edit_center[0] = self.edit_center[0] - 0.1
                stdscr.addstr(1, 10, f"Value of center after pressing 'a': {self.edit_center} ")
                stdscr.refresh()
            elif key == ord('d'):
                assert sum(self.recenter_edit_origin) == 0, "can not change center after recentered!"
                self.edit_center[0] = self.edit_center[0] + 0.1
                stdscr.addstr(1, 10, f"Value of center after pressing 'd': {self.edit_center}")
                stdscr.refresh()
            elif key == ord('w'):
                assert sum(self.recenter_edit_origin) == 0, "can not change center after recentered!"
                self.edit_center[2] = self.edit_center[2] + 0.1
                stdscr.addstr(1, 10, f"Value of center after pressing 'w': {self.edit_center}")
                stdscr.refresh()
            elif key == ord('x'):
                assert sum(self.recenter_edit_origin) == 0, "can not change center after recentered!"
                self.edit_center[2] = self.edit_center[2] - 0.1
                stdscr.addstr(1, 10, f"Value of center after pressing 'x': {self.edit_center}")
                stdscr.refresh()
            elif key == ord('z'):
                assert sum(self.recenter_edit_origin) == 0, "can not change center after recentered!"
                self.edit_center[1] = self.edit_center[1] - 0.1
                stdscr.addstr(1, 10, f"Value of center after pressing 'z': {self.edit_center}")
                stdscr.refresh()
            elif key == ord('e'):
                assert sum(self.recenter_edit_origin) == 0, "can not change center after recentered!"
                self.edit_center[1] = self.edit_center[1] + 0.1
                stdscr.addstr(1, 10, f"Value of center after pressing 'e': {self.edit_center}")
                stdscr.refresh()
            # angle min
            elif key == curses.KEY_UP:
                self.cfg.edit_elevation_range[0] = self.cfg.edit_elevation_range[0] + 10
                # self.cfg.edit_elevation_range[1] = self.cfg.edit_elevation_range[1] + 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'up arrow': {self.cfg.edit_elevation_range}")
                stdscr.refresh()
            elif key == curses.KEY_DOWN:
                self.cfg.edit_elevation_range[0] = self.cfg.edit_elevation_range[0] - 10
                # self.cfg.edit_elevation_range[1] = self.cfg.edit_elevation_range[1] - 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'down arrow': {self.cfg.edit_elevation_range}")
                stdscr.refresh()
            elif key == curses.KEY_LEFT:
                self.cfg.edit_azimuth_range[0] = self.cfg.edit_azimuth_range[0] - 10
                # self.cfg.edit_azimuth_range[1] = self.cfg.edit_azimuth_range[1] - 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'left arrow': {self.cfg.edit_azimuth_range}")
                stdscr.refresh()
            elif key == curses.KEY_RIGHT:
                self.cfg.edit_azimuth_range[0] = self.cfg.edit_azimuth_range[0] + 10
                # self.cfg.edit_azimuth_range[1] = self.cfg.edit_azimuth_range[1] + 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'right arrow': {self.cfg.edit_azimuth_range}")
                stdscr.refresh()
            # angle max
            elif key == ord('i'):
                # self.cfg.edit_elevation_range[0] = self.cfg.edit_elevation_range[0] + 10
                self.cfg.edit_elevation_range[1] = self.cfg.edit_elevation_range[1] + 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'up arrow': {self.cfg.edit_elevation_range}")
                stdscr.refresh()
            elif key == ord(','):
                # self.cfg.edit_elevation_range[0] = self.cfg.edit_elevation_range[0] - 10
                self.cfg.edit_elevation_range[1] = self.cfg.edit_elevation_range[1] - 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'down arrow': {self.cfg.edit_elevation_range}")
                stdscr.refresh()
            elif key == ord('j'):
                # self.cfg.edit_azimuth_range[0] = self.cfg.edit_azimuth_range[0] - 10
                self.cfg.edit_azimuth_range[1] = self.cfg.edit_azimuth_range[1] - 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'left arrow': {self.cfg.edit_azimuth_range}")
                stdscr.refresh()
            elif key == ord('l'):
                # self.cfg.edit_azimuth_range[0] = self.cfg.edit_azimuth_range[0] + 10
                self.cfg.edit_azimuth_range[1] = self.cfg.edit_azimuth_range[1] + 10
                stdscr.addstr(1, 10, f"Value of edit elevation after pressing 'right arrow': {self.cfg.edit_azimuth_range}")
                stdscr.refresh()
            # r
            elif key == ord('+'):
                self.cfg.edit_camera_distance_range[0] = self.cfg.edit_camera_distance_range[0] + 0.1
                self.cfg.edit_camera_distance_range[1] = self.cfg.edit_camera_distance_range[1] + 0.1
                stdscr.addstr(1, 10, f"Value of r after pressing '+': {self.cfg.edit_camera_distance_range}")
                stdscr.refresh()
            elif key == ord('-'):
                self.cfg.edit_camera_distance_range[0] = self.cfg.edit_camera_distance_range[0] - 0.1
                self.cfg.edit_camera_distance_range[1] = self.cfg.edit_camera_distance_range[1] - 0.1
                stdscr.addstr(1, 10, f"Value of r after pressing '-': {self.cfg.edit_camera_distance_range}")
                stdscr.refresh()
            elif key == ord('r'):
                self.recenter_edit_camera()
                stdscr.addstr(1, 10, f"recenter after pressing 'r': {self.cfg.edit_camera_distance_range}")
                stdscr.refresh()
            # add part
            elif key == ord('t'):
                self.add_edit_part()
                stdscr.addstr(1, 10, f"add edit part after pressing 'r': {self.edit_prompt}")
                stdscr.refresh()
            # input edit prompt
            if key == ord(' '):  # space key pressed
                stdscr.addstr(1, 10, "Enter a string (End with Enter key): ")
                input_str = ""
                while True:
                    char = stdscr.getch()
                    if char == curses.KEY_ENTER or char == 10:  # Enter key pressed
                        break
                    elif char == curses.KEY_BACKSPACE or char == 8 or char == 127:  # Backspace
                        input_str = input_str[:-1]
                        stdscr.addch(8, curses.A_NORMAL)  # Move cursor one step back
                        stdscr.addch(' ', curses.A_NORMAL)
                        stdscr.addch(8, curses.A_NORMAL)  # Move cursor one step back again
                    else:
                        input_str += chr(char)
                        stdscr.addch(char)
                self.edit_prompt = input_str
                stdscr.addstr(2, 10, f"You entered: {self.edit_prompt}")
            # exit
            elif key == ord('c'):
                self.edit_mode = False
                stdscr.addstr(1, 10, "...... END edit mode ......")
                stdscr.refresh()
                break
            elif key == ord('q'):
                stdscr.addstr(1, 10, "Exiting interactive mode.              ")
                stdscr.refresh()
                break
            else:
                stdscr.addstr(1, 10, f"Unknown key: {chr(key)}. Press 'a d, w x, z e, or q'...      ")
                stdscr.refresh()
    
    def modify_param(self):
        curses.wrapper(self._interactive_mode)
    
    def recenter_edit_camera(self):
        elevation = (self.cfg.edit_elevation_range[0] + self.cfg.edit_elevation_range[1]) / 2.
        azimuth = (self.cfg.edit_azimuth_range[0] + self.cfg.edit_azimuth_range[1]) / 2.
        camera_distances = (self.cfg.edit_camera_distance_range[0] + self.cfg.edit_camera_distance_range[1]) / 2.
        cur_position = torch.tensor([camera_distances * math.cos(elevation) * math.cos(azimuth),
                                     camera_distances * math.cos(elevation) * math.sin(azimuth),
                                     camera_distances * math.sin(elevation),
                                     ]) # [3]
        new_camera_distances = cur_position - torch.tensor(self.edit_center)
        new_camera_distances = torch.sqrt(torch.square(new_camera_distances).sum())
        self.cfg.edit_camera_distance_range[0] = new_camera_distances.item()
        self.cfg.edit_camera_distance_range[1] = new_camera_distances.item()
        self.recenter_edit_origin = self.edit_center
    
    def move_camera(self):
        if not self.edit_mode:
            if self.save_edit_flag:
                self.save_edit()
                self.save_edit_flag = False
                print("====== ", self.edit_prompt_list, " ==========")
            self.random_edit()
        edit_center = self.edit_center

        # edit_batch_size = self.cfg.edit_bs
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        real_batch_size = self.cfg.edit_bs // self.cfg.edit_n_view
        # if random.random() < 0.5:
        if random.random() < 1.: # TODO: refactor this
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.cfg.edit_elevation_range[1] - self.cfg.edit_elevation_range[0])
                + self.cfg.edit_elevation_range[0]
            ).repeat_interleave(self.cfg.edit_n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.cfg.edit_elevation_range[0] + 90.0) / 180.0,
                (self.cfg.edit_elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.edit_n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1,1) + torch.arange(self.cfg.edit_n_view).reshape(1,-1)
        ).reshape(-1) / self.cfg.edit_n_view * (
            self.cfg.edit_azimuth_range[1] - self.cfg.edit_azimuth_range[0]
        ) + self.cfg.edit_azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.edit_fovy_range[1] - self.cfg.edit_fovy_range[0])
            + self.cfg.edit_fovy_range[0]
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.edit_camera_distance_range[1] - self.cfg.edit_camera_distance_range[0])
            + self.cfg.edit_camera_distance_range[0]
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)
        if self.cfg.edit_relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth) + self.recenter_edit_origin[0],
                camera_distances * torch.cos(elevation) * torch.sin(azimuth) + self.recenter_edit_origin[1],
                camera_distances * torch.sin(elevation) + self.recenter_edit_origin[2],
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions) + torch.tensor(edit_center)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.edit_bs, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.edit_n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(self.cfg.edit_n_view, dim=0) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        else:
            raise NotImplementedError

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.cfg.edit_height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal_edit[
            None, :, :, :
        ].repeat(self.cfg.edit_bs, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.edit_width / self.cfg.edit_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        edit_batch = {
            'rays_o': rays_o,
            'rays_d': rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "edit_prompt": self.edit_prompt,
            "height": self.cfg.edit_height,
            "width": self.cfg.edit_width,
        }

        return edit_batch



    def collate(self, batch) -> Dict[str, Any]:
        assert self.batch_size % self.cfg.n_view == 0, f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        self.cnt += 1
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1,1) + torch.arange(self.cfg.n_view).reshape(1,-1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth = azimuth_deg * math.pi / 180


        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy_deg * math.pi / 180

        if self.cfg.fix_fov and self.init_cnt < self.cfg.init_steps:
            if self.init_cnt == 0:
                print(f"====ATTENTION: fix fov to {self.fov_value} before {self.cfg.init_steps}====")
            self.init_cnt += 1
            fovy = fovy * 0. + self.fov_value ########## NOTE(lihe) ##########
            fovy_deg = fovy / math.pi * 180
        
        if self.fix_view: # NOTE: fix 8 views
            print("++++++++++++++++ FIX VIEWs ++++++++++++++++")
            # elevation
            elevation = elevation * 0. + math.pi / 6.
            elevation_deg = 30
            # azimuth
            azimuth_deg = (
                torch.arange(self.cfg.n_view).reshape(1,-1)
            ).reshape(-1) / self.cfg.n_view * (
                self.azimuth_range[1] - self.azimuth_range[0]
            ) + self.azimuth_range[
                0
            ]
            azimuth = azimuth_deg * math.pi / 180
            # fov
            fovy = fovy * 0. + 0.8575560450553894 # TODO: refactor this
            fovy_deg = fovy / math.pi * 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        if self.cfg.relative_radius and not self.cfg.disable_relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_positions = camera_positions + camera_perturb
        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(self.cfg.n_view, dim=0) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(self.cfg.n_view, dim=0)  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(self.cfg.n_view, dim=0)  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0
        # shape init
        # NOTE: generate camera
        if self.shape_init and self.init_cnt < self.cfg.init_steps:
            # gsgen camera
            focal = 0.5 / torch.tan(fovy / 2) * self.height
            camera_info = CameraInfo(
                    focal[0], # focal[0],
                    focal[0], #focal[0],
                    self.height / 2.0,
                    self.width / 2.0,
                    self.height,
                    self.width,
                    0.01, # self.near_plane,
                    100.0, # self.far_plane,
                )
            # camera_info = [camera_info] * self.cfg.n_view
            camera_info = [camera_info] * self.batch_size
            c2w_gsgen = torch.cat(
                [torch.stack([right, -up, lookat], dim=-1), camera_positions[:, :, None]],
                dim=-1,
            )
            # c2w_lift3d = torch.cat(
            #     [torch.stack([-up, right, -lookat], dim=-1), camera_positions[:, :, None]],
            #     dim=-1,
            # )
            camera_positions_lift3d = camera_positions / camera_distances.unsqueeze(1) * 1.2   # b, 3
            c2w_lift3d = torch.cat(
                # [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
                [torch.stack([right, up, -lookat], dim=-1), camera_positions_lift3d[:, :, None]],
                dim=-1,
            )
            
            sampled_cameras = {
                # "pos": pos,
                "c2w": c2w_gsgen,
                "c2w_lift3d": c2w_lift3d,
                "camera_info": camera_info,
                "elevation": elevation * 180 / math.pi,
                "azimuth": azimuth.type(torch.float64) * 180 / math.pi,
                "camera_distance": camera_distances.type(torch.float64),
                "light_pos": light_positions,
                "light_color": torch.ones(self.cfg.n_view, 3),
            }
        else:
            sampled_cameras = None

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )
        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        ##### start modify editing camera poses #####
        # if self.cfg.enable_edit:
        if self.cfg.enable_edit and self.cnt >= self.cfg.start_edit_step:
            if self.edit_mode:
                print("...... Enter edit mode .....")
                self.modify_param()
            edit_batch = self.move_camera()
        else:
            edit_batch = None

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
            "sampled_cameras": sampled_cameras,
            "edit_batch": edit_batch,
            "fix_view": self.fix_view,
        }


@register("gs-edit-multiview-camera-datamodule")
class RandomMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: GSEditRandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(GSEditRandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GSEditRandomMultiviewCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
