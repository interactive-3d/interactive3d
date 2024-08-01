from dataclasses import dataclass

import nerfacc
import torch
import torch.nn.functional as F

import os
import threestudio
from threestudio.models.background.base import BaseBackground
from threestudio.models.geometry.base import BaseImplicitGeometry
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.renderers.base import VolumeRenderer
from threestudio.utils.ops import chunk_batch, validate_empty_rays
from threestudio.utils.typing import *
import numpy as np
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)


@threestudio.register("nerf-volume-renderer")
class NeRFVolumeRenderer(VolumeRenderer):
    @dataclass
    class Config(VolumeRenderer.Config):
        num_samples_per_ray: int = 512
        randomized: bool = True
        eval_chunk_size: int = 160000
        grid_prune: bool = True
        prune_alpha_threshold: bool = True
        return_comp_normal: bool = False
        return_normal_perturb: bool = False
        alpha_thre: float = 0.01
        edit: bool = False
        index_str: str = ""
        mask_occ: bool = False

    cfg: Config

    def configure(
        self,
        geometry: BaseImplicitGeometry,
        material: BaseMaterial,
        background: BaseBackground,
    ) -> None:
        super().configure(geometry, material, background)
        # introduce edit part
        self.edit = self.cfg.edit
        self.save_occ = self.cfg.get('save_occ', False)
        if not self.edit:
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
        else:
            # global estimator, not used
            self.estimator = nerfacc.OccGridEstimator(
                roi_aabb=self.bbox.view(-1), resolution=32, levels=1
            )
            if self.cfg.mask_occ: # no use now
                self.mask0 = np.load('debug_data/occ_index_0_gs_mask.npy')
                self.mask1 = np.load('debug_data/occ_index_1_gs_mask.npy')
            if not self.cfg.grid_prune:
                self.estimator.occs.fill_(True)
                self.estimator.binaries.fill_(True)
            
            # part estimator
            self.occ_index_list = []
            part_num = self.geometry.part_num
            print("====index str====", self.cfg.index_str)
            if part_num == 1:
                print("======only have one part, all true========")
                occ_index = self.estimator.binaries.view(-1) >= 0
                occ_index = occ_index.nonzero().squeeze(1)
                self.occ_index_list.append(occ_index)
            else:
                for i in range(part_num):
                    occ_index = torch.from_numpy(np.load(f'debug_data/occ_index_{i}' + self.cfg.index_str + '.npy')).long()
                    self.occ_index_list.append(occ_index)
                    
            # NOTE: for mesh exporter
            self.geometry.occ_index_list = self.occ_index_list

        self.render_step_size = (
            1.732 * 2 * self.cfg.radius / self.cfg.num_samples_per_ray
        )
        self.randomized = self.cfg.randomized
    
    def custmoize_sample(self, rays_o_flatten, rays_d_flatten, sigma_fn=None, render_step_size=None,
                         alpha_thre=0.0, stratified=None, cone_angle=0.0, early_stop_eps=0,):
        render_step_size = self.render_step_size
        stratified = self.randomized
        rays_o = rays_o_flatten
        rays_d = rays_d_flatten
        near_plane: float = 0.0
        far_plane: float = 1e10
        t_min = None
        t_max = None
        assert sigma_fn is not None
        # The following code are adapted from nerfacc
        near_planes = torch.full_like(rays_o[..., 0], fill_value=near_plane)
        far_planes = torch.full_like(rays_o[..., 0], fill_value=far_plane)

        if t_min is not None:
            near_planes = torch.clamp(near_planes, min=t_min)
        if t_max is not None:
            far_planes = torch.clamp(far_planes, max=t_max)

        if stratified:
            near_planes += torch.rand_like(near_planes) * render_step_size
        # intervals, samples, _ = nerfacc.grid.traverse_grids(
        intervals, samples = nerfacc.grid.traverse_grids(
            rays_o,
            rays_d,
            self.estimator.binaries,
            self.estimator.aabbs,
            near_planes=near_planes,
            far_planes=far_planes,
            step_size=render_step_size,
            cone_angle=cone_angle,
        )
        t_starts = intervals.vals[intervals.is_left]
        t_ends = intervals.vals[intervals.is_right]
        ray_indices = samples.ray_indices
        packed_info = samples.packed_info

        alpha_thre = min(alpha_thre, self.estimator.occs.mean().item())
        # Compute visibility of the samples, and filter out invisible samples
        if sigma_fn is not None:
            if t_starts.shape[0] != 0:
                sigmas, only_train_mask = sigma_fn(t_starts, t_ends, ray_indices)
            else:
                sigmas = torch.empty((0,), device=t_starts.device)
            assert (
                sigmas.shape == t_starts.shape
            ), "sigmas must have shape of (N,)! Got {}".format(sigmas.shape)
            masks = nerfacc.volrend.render_visibility_from_density(
                t_starts=t_starts,
                t_ends=t_ends,
                sigmas=sigmas,
                packed_info=packed_info,
                early_stop_eps=early_stop_eps,
                alpha_thre=alpha_thre,
            )

            # NOTE: dont change the only train part grids
            if len(sigmas) > 0:
                masks = torch.logical_or(masks, only_train_mask)
        
        ray_indices, t_starts, t_ends = (
                ray_indices[masks],
                t_starts[masks],
                t_ends[masks],
            )
        return ray_indices, t_starts, t_ends

    def forward(
        self,
        rays_o: Float[Tensor, "B H W 3"],
        rays_d: Float[Tensor, "B H W 3"],
        light_positions: Float[Tensor, "B 3"],
        bg_color: Optional[Tensor] = None,
        super_hash: Optional[bool] = False,
        render_org: Optional[bool] = False,
        force_shading: Optional[str] = None,
        force_diffuse_light_color: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Float[Tensor, "..."]]:
        batch_size, height, width = rays_o.shape[:3]
        rays_o_flatten: Float[Tensor, "Nr 3"] = rays_o.reshape(-1, 3)
        rays_d_flatten: Float[Tensor, "Nr 3"] = rays_d.reshape(-1, 3)
        light_positions_flatten: Float[Tensor, "Nr 3"] = (
            light_positions.reshape(-1, 1, 1, 3)
            .expand(-1, height, width, -1)
            .reshape(-1, 3)
        )
        n_rays = rays_o_flatten.shape[0]

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions
            if self.training:
                sigma = self.geometry.forward_density(positions)[..., 0]
            else:
                sigma = chunk_batch(
                    self.geometry.forward_density,
                    self.cfg.eval_chunk_size,
                    positions,
                )[..., 0]
            return sigma
        
        def sigma_fn_super(t_starts, t_ends, ray_indices):
            t_starts, t_ends = t_starts[..., None], t_ends[..., None]
            t_origins = rays_o_flatten[ray_indices]
            t_positions = (t_starts + t_ends) / 2.0
            t_dirs = rays_d_flatten[ray_indices]
            positions = t_origins + t_dirs * t_positions

            with torch.no_grad():
                semantic_features = torch.zeros_like(self.estimator.binaries).view(-1).float() - 1. # [1, 32, 32, 32]
                # semantic_features = torch.zeros_like(self.estimator.binaries).view(-1).float() # [1, 32, 32, 32]
                for i in range(self.geometry.part_num):
                    occ_index = self.occ_index_list[i]
                    semantic_features[occ_index] = i
                semantic_features = semantic_features.view(*self.estimator.binaries.shape)
                semantic_features = semantic_features.unsqueeze(1) # [1, 1, 32, 32, 32]
                points_norm = contract_to_unisphere(positions, self.geometry.bbox, self.geometry.unbounded) # [0, 1]
                # points_norm = contract_to_unisphere(positions, norm_bbox.to(self.geometry.bbox.device), self.geometry.unbounded) # [0, 1]
                points_norm = points_norm * 2 - 1. # [-1, 1]
                points_norm = points_norm.view(1, 1, 1, points_norm.shape[0], 3) # [b, h, w, d, 3]
                points_norm_flip = torch.flip(points_norm, dims=[-1])
                part_flag = F.grid_sample(semantic_features, points_norm_flip, mode='nearest')
                part_flag = part_flag.view(-1)

                # NOTE: now we compute fine mask
                if self.training:
                    occ_index_fine = torch.from_numpy(np.load(f'debug_data/occ_index_0' + self.cfg.index_str + '_fine' + '.npy')).long()
                    semantic_features = semantic_features.view(-1) * 0. - 1.
                    semantic_features[occ_index_fine] = 0
                    semantic_features = semantic_features.view(*self.estimator.binaries.shape).unsqueeze(1)
                    fine_flag = F.grid_sample(semantic_features, points_norm_flip, mode='nearest')
                    fine_flag = fine_flag.view(-1)
            
            # print("===debug part flag===")
            # np.save('debug_data/part_flag2.npy', part_flag.detach().cpu().numpy())
            # np.save('debug_data/points_norm2.npy', points_norm.detach().cpu().numpy())
            # np.save('debug_data/positions2.npy', positions.detach().cpu().numpy())

            if self.training:
                sigma = self.geometry.forward_density_super(positions, part_flag)[..., 0]
            else:
                sigma = chunk_batch(
                    self.geometry.forward_density_super,
                    self.cfg.eval_chunk_size,
                    positions,
                    part_flag,
                )[..., 0]
            # NOTE: we dont prune the training part points
            # NOTE: we use smaller fine-tune mask

            only_train_mask = torch.zeros_like(part_flag).type(torch.bool) # all false
            if self.training:
                # use fine mask
                only_train_mask = fine_flag == 0
                # for i in self.geometry.cfg.only_train_part:
                #     inds = part_flag == i
                #     only_train_mask = torch.logical_or(only_train_mask, inds)
                    # sigma[inds] = torch.clip(sigma[inds], 1.0)

            return sigma, only_train_mask

        if not self.cfg.grid_prune:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=self.randomized,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
        else:
            with torch.no_grad():
                if not self.edit:
                    ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                        render_step_size=self.render_step_size,
                        alpha_thre=self.cfg.alpha_thre if self.cfg.prune_alpha_threshold else 0.0,
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )
                else:
                    # customize sampling
                    # ray_indices, t_starts_, t_ends_ = self.custmoize_sample(
                    #     rays_o_flatten,
                    #     rays_d_flatten,
                    #     sigma_fn=sigma_fn_super, 
                    #     render_step_size=self.render_step_size,
                    #     alpha_thre=self.cfg.alpha_thre, 
                    #     stratified=self.randomized,
                    #     cone_angle=0.0,
                    # )
                    global_ray_indices, global_t_starts_, global_t_ends_ = self.estimator.sampling(
                        rays_o_flatten,
                        rays_d_flatten,
                        sigma_fn=sigma_fn if self.cfg.prune_alpha_threshold else None,
                        # sigma_fn=sigma_fn_super, 
                        render_step_size=self.render_step_size,
                        alpha_thre=self.cfg.alpha_thre if self.cfg.prune_alpha_threshold else 0.0,
                        # alpha_thre=self.cfg.alpha_thre, 
                        stratified=self.randomized,
                        cone_angle=0.0,
                    )
                    ray_indices, t_starts_, t_ends_ = global_ray_indices, global_t_starts_, global_t_ends_
                    
        ray_indices, t_starts_, t_ends_ = validate_empty_rays(
            ray_indices, t_starts_, t_ends_
        )
        ray_indices = ray_indices.long()
        t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_light_positions = light_positions_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        # use grid sample to find semantic label
        if self.edit:
            with torch.no_grad():
                semantic_features = torch.zeros_like(self.estimator.binaries).view(-1).float() - 1. # [1, 32, 32, 32]
                for i in range(self.geometry.part_num):
                    occ_index = self.occ_index_list[i]
                    semantic_features[occ_index] = i
                semantic_features = semantic_features.view(*self.estimator.binaries.shape)
                semantic_features = semantic_features.unsqueeze(1) # [1, 1, 32, 32, 32]
                # NOTE: dont use bbox -1 ~ 1
                # norm_bbox = torch.as_tensor(
                #         [
                #             [-1., -1., -1.],
                #             [1., 1., 1.],
                #         ],
                #         dtype=torch.float32,
                #     )
                points_norm = contract_to_unisphere(positions, self.geometry.bbox, self.geometry.unbounded) # [0, 1]
                points_norm = points_norm * 2 - 1. # [-1, 1]
                points_norm = points_norm.view(1, 1, 1, points_norm.shape[0], 3) # [b, h, w, d, 3]
                points_norm_flip = torch.flip(points_norm, dims=[-1])
                part_flag = F.grid_sample(semantic_features, points_norm_flip, mode='nearest')
                part_flag = part_flag.view(-1)
                # print("===debug save part flag===")
                # np.save('debug_data/part_flag.npy', part_flag.detach().cpu().numpy())
                # np.save('debug_data/points_norm.npy', points_norm.detach().cpu().numpy())
                # np.save('debug_data/positions.npy', positions.detach().cpu().numpy())

        if self.training:
            geo_out = self.geometry(
                positions, output_normal=self.material.requires_normal, super_hash=super_hash, part_flag=part_flag if self.edit else None,
                render_org=render_org,
            )
            rgb_fg_all = self.material(
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                shading=self.material.cur_shading if render_org else None,
                force_shading=force_shading,
                force_diffuse_light_color=force_diffuse_light_color,
                **geo_out,
                **kwargs
            )
            comp_rgb_bg = self.background(dirs=rays_d) # NOTE: check the background of residual and original renderings
        else:
            geo_out = chunk_batch(
                self.geometry,
                self.cfg.eval_chunk_size,
                positions,
                output_normal=self.material.requires_normal,
                super_hash=super_hash,
                part_flag=part_flag if self.edit else None
            )
            rgb_fg_all = chunk_batch(
                self.material,
                self.cfg.eval_chunk_size,
                viewdirs=t_dirs,
                positions=positions,
                light_positions=t_light_positions,
                **geo_out
            )
            comp_rgb_bg = chunk_batch(
                self.background, self.cfg.eval_chunk_size, dirs=rays_d
            )

        weights: Float[Tensor, "Nr 1"]
        weights_, _, _ = nerfacc.render_weight_from_density(
            t_starts[..., 0],
            t_ends[..., 0],
            geo_out["density"][..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )
        weights = weights_[..., None]
        opacity: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        depth: Float[Tensor, "Nr 1"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=t_positions, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_rgb_fg: Float[Tensor, "Nr Nc"] = nerfacc.accumulate_along_rays(
            weights[..., 0], values=rgb_fg_all, ray_indices=ray_indices, n_rays=n_rays
        )

        # populate depth and opacity to each point
        t_depth = depth[ray_indices]
        z_variance = nerfacc.accumulate_along_rays(
            weights[..., 0],
            values=(t_positions - t_depth) ** 2,
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        if bg_color is None:
            bg_color = comp_rgb_bg
        else:
            if bg_color.shape[:-1] == (batch_size,):
                # e.g. constant random color used for Zero123
                # [bs,3] -> [bs, 1, 1, 3]):
                bg_color = bg_color.unsqueeze(1).unsqueeze(1)
                #        -> [bs, height, width, 3]):
                bg_color = bg_color.expand(-1, height, width, -1)

        if bg_color.shape[:-1] == (batch_size, height, width):
            bg_color = bg_color.reshape(batch_size * height * width, -1)

        comp_rgb = comp_rgb_fg + bg_color * (1.0 - opacity)

        out = {
            "comp_rgb": comp_rgb.view(batch_size, height, width, -1),
            "comp_rgb_fg": comp_rgb_fg.view(batch_size, height, width, -1),
            "comp_rgb_bg": comp_rgb_bg.view(batch_size, height, width, -1),
            "opacity": opacity.view(batch_size, height, width, 1),
            "depth": depth.view(batch_size, height, width, 1),
            "z_variance": z_variance.view(batch_size, height, width, 1),
            "bg_color": bg_color.detach(), # NOTE: save bg color to render original
        }

        if self.training:
            out.update(
                {
                    "weights": weights,
                    "t_points": t_positions,
                    "t_intervals": t_intervals,
                    "t_dirs": t_dirs,
                    "ray_indices": ray_indices,
                    "points": positions,
                    **geo_out,
                }
            )
            if "normal" in geo_out:
                if self.cfg.return_comp_normal:
                    comp_normal: Float[Tensor, "Nr 3"] = nerfacc.accumulate_along_rays(
                        weights[..., 0],
                        values=geo_out["normal"],
                        ray_indices=ray_indices,
                        n_rays=n_rays,
                    )
                    comp_normal = F.normalize(comp_normal, dim=-1)
                    comp_normal = (
                        (comp_normal + 1.0) / 2.0 * opacity
                    )  # for visualization
                    out.update(
                        {
                            "comp_normal": comp_normal.view(
                                batch_size, height, width, 3
                            ),
                        }
                    )
                if self.cfg.return_normal_perturb:
                    normal_perturb = self.geometry(
                        positions + torch.randn_like(positions) * 1e-2,
                        output_normal=self.material.requires_normal,
                        super_hash=super_hash,
                    )["normal"]
                    out.update({"normal_perturb": normal_perturb})
        else:
            if "normal" in geo_out:
                comp_normal = nerfacc.accumulate_along_rays(
                    weights[..., 0],
                    values=geo_out["normal"],
                    ray_indices=ray_indices,
                    n_rays=n_rays,
                )
                comp_normal = F.normalize(comp_normal, dim=-1)
                comp_normal = (comp_normal + 1.0) / 2.0 * opacity  # for visualization
                out.update(
                    {
                        "comp_normal": comp_normal.view(batch_size, height, width, 3),
                    }
                )

        return out

    def update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ) -> None:
        if self.cfg.grid_prune:

            def occ_eval_fn(x):
                density = self.geometry.forward_density(x)
                # approximate for 1 - torch.exp(-density * self.render_step_size) based on taylor series
                return density * self.render_step_size

            if self.cfg.mask_occ:
                mask0 = self.mask0
                mask1 = self.mask1
                self.estimator.occs[mask0] = 0.
                self.estimator.occs[mask1] = 0.
                binaries = self.estimator.binaries.view(-1)
                binaries[mask0] = 0
                binaries[mask1] = 0
                self.estimator.binaries = binaries.view(*self.estimator.binaries.shape)
                print("=======pruning floater occs==========")

            if self.save_occ: # save occ for selecting interested regions
                print("==== save occ and exit ===")
                binaries = self.estimator.binaries
                occs = self.estimator.occs
                grid_coords = self.estimator.grid_coords
                os.makedirs('debug_data', exist_ok=True)
                np.save('debug_data/binaries.npy', binaries.detach().cpu().numpy())
                np.save('debug_data/occs.npy', occs.detach().cpu().numpy())
                np.save('debug_data/grid_coords.npy', grid_coords.detach().cpu().numpy())
                print("====binaries===", binaries.sum(), binaries.shape)
                print("====occs===", occs.sum(), occs.shape)
                print("====occs min===", occs.min(), occs.shape)
                print("====grid_coords===", grid_coords.shape)
                exit()

            if self.training and not on_load_weights and not self.edit: # NOTE: disable occ update if editting
                self.estimator.update_every_n_steps(
                    step=global_step, occ_eval_fn=occ_eval_fn
                )
                # NOTE: load part information
                # part_num = self.geometry.part_num
                # occ_index_list = []
                # for i in range(part_num):
                #     occ_index = np.load(f'debug_data/occ_index_{i}.npy')
                #     occ_index = torch.from_numpy(occ_index)
                #     occ_index_list.append(occ_index)

    def train(self, mode=True):
        self.randomized = mode and self.cfg.randomized
        return super().train(mode=mode)

    def eval(self):
        self.randomized = False
        return super().eval()
