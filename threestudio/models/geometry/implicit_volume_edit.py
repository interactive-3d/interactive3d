from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseGeometry,
    BaseImplicitGeometry,
    contract_to_unisphere,
)
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import get_activation
from threestudio.utils.typing import *


@threestudio.register("implicit-volume-edit")
class ImplicitVolumeEdit(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        density_activation: Optional[str] = "softplus"
        density_bias: Union[float, str] = "blob_magic3d"
        density_blob_scale: float = 10.0
        density_blob_std: float = 0.5
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        # edit
        edit_pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 4096,
                "per_level_scale": 2,
            }
        )
        edit_mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        # end edit
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: float = 0.01

        # automatically determine the threshold
        isosurface_threshold: Union[float, str] = 25.0

        # part num
        part_num: int = 0
        only_train_part: Any = -1

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.density_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        )
        # edit
        if self.cfg.part_num > 0:
            self.part_num = self.cfg.part_num
            self.edit_encoding_list = []
            self.edit_density_network_list = []
            self.edit_feature_network_list = []
            for i in range(self.part_num):
                edit_encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.edit_pos_encoding_config
                )
                edit_density_network = get_mlp(
                    edit_encoding.n_output_dims + self.encoding.n_output_dims + 1, 
                    # self.edit_encoding.n_output_dims, 
                    1, self.cfg.edit_mlp_network_config
                )
                self.edit_encoding_list.append(edit_encoding)
                self.edit_density_network_list.append(edit_density_network)
        else:
            self.edit_encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.edit_pos_encoding_config
            )
            self.edit_density_network = get_mlp(
                self.edit_encoding.n_output_dims + self.encoding.n_output_dims + 1, 
                # self.edit_encoding.n_output_dims, 
                1, self.cfg.edit_mlp_network_config
            )
        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
            if self.cfg.part_num > 0:
                for i in range(self.part_num):
                    edit_feature_network = get_mlp(
                        self.edit_encoding_list[i].n_output_dims + self.encoding.n_output_dims + self.cfg.n_feature_dims,
                        # self.edit_encoding.n_output_dims,
                        self.cfg.n_feature_dims,
                        self.cfg.edit_mlp_network_config,
                    )
                    self.edit_feature_network_list.append(edit_feature_network)
            else:
                self.edit_feature_network = get_mlp(
                self.edit_encoding.n_output_dims + self.encoding.n_output_dims + self.cfg.n_feature_dims,
                # self.edit_encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.edit_mlp_network_config,)

        if self.cfg.part_num > 0:
            self.edit_encoding_list = nn.ModuleList(self.edit_encoding_list)
            self.edit_density_network_list = nn.ModuleList(self.edit_density_network_list)
            self.edit_feature_network_list = nn.ModuleList(self.edit_feature_network_list)
            # self.occ_index_list = []
            # for i in range(self.cfg.part_num):
            #     occ_index = torch.from_numpy(np.load(f'debug_data/occ_index_{i}.npy')).long()
            #     self.occ_index_list.append(occ_index)

        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        
        self.set_fine_tune_mode() # NOTE(lihe): not fine-tune

    
    def set_fine_tune_mode(self):
        self.encoding.requires_grad_(False)
        self.density_network.requires_grad_(False)
        self.feature_network.requires_grad_(False)
        if isinstance(self.cfg.only_train_part, int):
            only_train_part = [self.cfg.only_train_part]
        else:
            only_train_part = list(self.cfg.only_train_part)
        print("Only Train Interested Region:", only_train_part)
        if len(only_train_part) >= 0:
            for i in range(self.cfg.part_num):
                if i not in only_train_part:
                    self.edit_encoding_list[i].requires_grad_(False)
                    self.edit_density_network_list[i].requires_grad_(False)
                    self.edit_feature_network_list[i].requires_grad_(False)

    def get_activated_density(
        self, points: Float[Tensor, "*N Di"], density: Float[Tensor, "*N 1"]
    ) -> Tuple[Float[Tensor, "*N 1"], Float[Tensor, "*N 1"]]:
        density_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.density_bias == "blob_dreamfusion":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * torch.exp(
                    -0.5 * (points**2).sum(dim=-1) / self.cfg.density_blob_std**2
                )[..., None]
            )
        elif self.cfg.density_bias == "blob_magic3d":
            # pre-activation density bias
            density_bias = (
                self.cfg.density_blob_scale
                * (
                    1
                    - torch.sqrt((points**2).sum(dim=-1)) / self.cfg.density_blob_std
                )[..., None]
            )
        elif isinstance(self.cfg.density_bias, float):
            density_bias = self.cfg.density_bias
        else:
            raise ValueError(f"Unknown density bias {self.cfg.density_bias}")
        raw_density: Float[Tensor, "*N 1"] = density + density_bias
        density = get_activation(self.cfg.density_activation)(raw_density)
        return raw_density, density

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False, super_hash: bool = False, part_flag = None, render_org=False,
    ) -> Dict[str, Float[Tensor, "..."]]:
        assert part_flag is not None
        #TODO: hard code, refactor this func
        super_hash = True
        if render_org:
            super_hash = False
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        density = self.density_network(enc).view(*points.shape[:-1], 1)
        
        if not super_hash:
            raw_density, density = self.get_activated_density(points_unscaled, density)
        else:
            if self.cfg.part_num > 0:
                select_inds_list = []
                select_edit_inp_list = []
                edit_density = torch.zeros_like(density)
                for i in range(self.part_num):
                    select_inds = part_flag == i
                    select_inds_list.append(select_inds)
                    select_points = points.view(-1, self.cfg.n_input_dims)[select_inds]
                    
                    select_edit_enc = self.edit_encoding_list[i](select_points)
                    select_enc = enc[select_inds]
                    select_density = density[select_inds]
                    select_edit_inp = torch.cat([select_enc, select_edit_enc], dim=-1)
                    select_edit_inp_list.append(select_edit_inp)
                    select_edit_inp_density = torch.cat([select_edit_inp, select_density], dim=-1)
                    select_edit_density = self.edit_density_network_list[i](select_edit_inp_density).view(*select_points.shape[:-1], 1)
                    # save density
                    edit_density[select_inds] = select_edit_density
                edit_density = density + edit_density
                raw_edit_density, density = self.get_activated_density(points_unscaled, edit_density)
            else:
                edit_enc = self.edit_encoding(points.view(-1, self.cfg.n_input_dims))
                edit_enc_inp = torch.cat([enc, edit_enc], dim=-1)
                edit_enc_inp_density = torch.cat([edit_enc_inp, density], dim=-1)
                edit_density = self.edit_density_network(edit_enc_inp_density).view(*points.shape[:-1], 1)
                edit_density = density + edit_density
                raw_edit_density, density = self.get_activated_density(points_unscaled, edit_density)

        output = {
            "density": density,
        }

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            if not super_hash:
                output.update({"features": features})
            else:
                if self.cfg.part_num > 0:
                    edit_features = torch.zeros_like(features)
                    for i in range(self.part_num):
                        select_inds = select_inds_list[i]
                        select_edit_enc_inp = select_edit_inp_list[i]
                        select_edit_inp_feature = torch.cat([select_edit_enc_inp, features[select_inds]], dim=-1)
                        select_edit_features = self.edit_feature_network_list[i](select_edit_inp_feature).view(*select_edit_inp_feature.shape[:-1], self.cfg.n_feature_dims)
                        edit_features[select_inds] = select_edit_features
                    # features = features + edit_features
                    features = features + edit_features
                else:
                    edit_enc_inp_feature = torch.cat([edit_enc_inp, features], dim=-1)
                    # edit_enc_inp_feature = edit_enc_inp
                    edit_features = self.edit_feature_network(edit_enc_inp_feature).view(
                        *points.shape[:-1], self.cfg.n_feature_dims
                    )
                    features = features + edit_features
                output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                # TODO: use raw density
                eps = self.cfg.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 6 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (
                        -0.5
                        * (density_offset[..., 0::2, 0] - density_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    density_offset: Float[Tensor, "... 3 1"] = self.forward_density(
                        points_offset
                    )
                    normal = (density_offset[..., 0::1, 0] - density) / eps
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
            elif self.cfg.normal_type == "analytic":
                normal = -torch.autograd.grad(
                    density,
                    points_unscaled,
                    grad_outputs=torch.ones_like(density),
                    create_graph=True,
                )[0]
                normal = F.normalize(normal, dim=-1)
                if not grad_enabled:
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update({"normal": normal, "shading_normal": normal})

        torch.set_grad_enabled(grad_enabled)
        return output

    def forward_density(self, points: Float[Tensor, "*N Di"], super_hash=False) -> Float[Tensor, "*N 1"]:
        # NOTE(lihe): hard code
        super_hash = False
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        density = self.density_network(
            enc
        ).reshape(*points.shape[:-1], 1)

        if not super_hash:
            _, density = self.get_activated_density(points_unscaled, density)
        else:
            edit_enc = self.edit_encoding(points.view(-1, self.cfg.n_input_dims))
            edit_enc_inp = torch.cat([enc, edit_enc, density], dim=-1)
            edit_density = self.edit_density_network(edit_enc_inp).view(*points.shape[:-1], 1)
            edit_density = density + edit_density
            _, density = self.get_activated_density(points_unscaled, edit_density)
        return density
    
    def forward_density_super(self, points: Float[Tensor, "*N Di"], part_flag=None) -> Float[Tensor, "*N 1"]:
        # NOTE(lihe): hard code
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        assert part_flag is not None
        assert self.cfg.part_num > 0
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        density = self.density_network(
            enc
        ).reshape(*points.shape[:-1], 1)

        select_inds_list = []
        select_edit_inp_list = []
        edit_density = torch.zeros_like(density)
        for i in range(self.part_num):
            select_inds = part_flag == i
            select_inds_list.append(select_inds)
            select_points = points.view(-1, self.cfg.n_input_dims)[select_inds]
            
            select_edit_enc = self.edit_encoding_list[i](select_points)
            select_enc = enc[select_inds]
            select_density = density[select_inds]
            select_edit_inp = torch.cat([select_enc, select_edit_enc], dim=-1)
            select_edit_inp_list.append(select_edit_inp)
            select_edit_inp_density = torch.cat([select_edit_inp, select_density], dim=-1)
            select_edit_density = self.edit_density_network_list[i](select_edit_inp_density).view(*select_points.shape[:-1], 1)
            # save density
            edit_density[select_inds] = select_edit_density
        # edit_density = density + edit_density
        edit_density = density + edit_density
        raw_edit_density, density = self.get_activated_density(points_unscaled, edit_density)

        return density

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        if self.cfg.isosurface_deformable_grid:
            threestudio.warn(
                f"{self.__class__.__name__} does not support isosurface_deformable_grid. Ignoring."
            )
        # NOTE: for mesh exporter
        print("==============super hash for mesh exporter==========")
        part_flag = self.get_part_flag(points)
        density = self.forward_density_super(points, part_flag=part_flag)
        return density, None
    
    def get_part_flag(self, positions):
        with torch.no_grad():
            semantic_features = torch.zeros(1, 32, 32, 32).to(positions.device).view(-1).float() - 1. # [1, 32, 32, 32]
            for i in range(self.part_num):
                occ_index = self.occ_index_list[i]
                semantic_features[occ_index] = i
            semantic_features = semantic_features.view(1, 32, 32, 32)
            semantic_features = semantic_features.unsqueeze(1) # [1, 1, 32, 32, 32]
            # NOTE(lihe): dont use bbox -1 ~ 1
            # norm_bbox = torch.as_tensor(
            #         [
            #             [-1., -1., -1.],
            #             [1., 1., 1.],
            #         ],
            #         dtype=torch.float32,
            #     )
            points_norm = contract_to_unisphere(positions, self.bbox, self.unbounded) # [0, 1]
            # points_norm = contract_to_unisphere(positions, norm_bbox.to(self.geometry.bbox.device), self.geometry.unbounded) # [0, 1]
            points_norm = points_norm * 2 - 1. # [-1, 1]
            points_norm = points_norm.view(1, 1, 1, points_norm.shape[0], 3) # [b, h, w, d, 3]
            points_norm_flip = torch.flip(points_norm, dims=[-1])
            part_flag = F.grid_sample(semantic_features, points_norm_flip, mode='nearest')
            part_flag = part_flag.view(-1)
        
        return part_flag

    def forward_feature_super(self, points, part_flag):
        super_hash = True
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        density = self.density_network(enc).view(*points.shape[:-1], 1)
        # NOTE: introduce for loop
        assert self.cfg.part_num > 0
        if self.cfg.part_num > 0:
            select_inds_list = []
            select_edit_inp_list = []
            edit_density = torch.zeros_like(density)
            print("part num: ", self.part_num)
            for i in range(self.part_num):
                select_inds = part_flag == i
                select_inds_list.append(select_inds)
                select_points = points.view(-1, self.cfg.n_input_dims)[select_inds]
                
                select_edit_enc = self.edit_encoding_list[i](select_points)
                select_enc = enc[select_inds]
                select_density = density[select_inds]
                select_edit_inp = torch.cat([select_enc, select_edit_enc], dim=-1)
                select_edit_inp_list.append(select_edit_inp)
            
        output = {}
        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            if self.cfg.part_num > 0:
                edit_features = torch.zeros_like(features)
                for i in range(self.part_num):
                    select_inds = select_inds_list[i]
                    select_edit_enc_inp = select_edit_inp_list[i]
                    select_edit_inp_feature = torch.cat([select_edit_enc_inp, features[select_inds]], dim=-1)
                    select_edit_features = self.edit_feature_network_list[i](select_edit_inp_feature).view(*select_edit_inp_feature.shape[:-1], self.cfg.n_feature_dims)
                    edit_features[select_inds] = select_edit_features
                features = features + edit_features
            output.update({"features": features})
        return output


    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return -(field - threshold)

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        
        # NOTE: mesh exporter
        part_flag = self.get_part_flag(points.reshape(-1, self.cfg.n_input_dims))
        with torch.no_grad():
            feature_out = self.forward_feature_super(points.reshape(-1, self.cfg.n_input_dims), part_flag)
            features = feature_out['features']
            features = features.view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
        out.update(
            {
                "features": features,
            }
        )
        return out

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "ImplicitVolumeEdit":
        if isinstance(other, ImplicitVolumeEdit):
            instance = ImplicitVolumeEdit(cfg, **kwargs)
            instance.encoding.load_state_dict(other.encoding.state_dict())
            instance.density_network.load_state_dict(other.density_network.state_dict())
            if copy_net:
                if (
                    instance.cfg.n_feature_dims > 0
                    and other.cfg.n_feature_dims == instance.cfg.n_feature_dims
                ):
                    instance.feature_network.load_state_dict(
                        other.feature_network.state_dict()
                    )
                if (
                    instance.cfg.normal_type == "pred"
                    and other.cfg.normal_type == "pred"
                ):
                    instance.normal_network.load_state_dict(
                        other.normal_network.state_dict()
                    )
            return instance
        else:
            raise TypeError(
                f"Cannot create {ImplicitVolumeEdit.__name__} from {other.__class__.__name__}"
            )
