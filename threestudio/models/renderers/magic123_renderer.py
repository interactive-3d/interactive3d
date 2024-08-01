import torch
import torch.nn as nn
import sys
import threestudio
from threestudio.utils.typing import *

from threestudio.utils.config import ExperimentConfig, load_config
import numpy as np
from torchvision.utils import save_image
import pdb
from omegaconf import OmegaConf
from dataclasses import dataclass, field

@threestudio.register("magic123-renderer")
class Magic123Renderer(nn.Module):
    def __init__(self, ckpt=None):
        super(Magic123Renderer, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = "path_to_your_checkpoint.ckpt"
        cfg_path = "parsed.yaml"
        # parse YAML config to OmegaConf
        cfg: ExperimentConfig
        # cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)
        cfg = load_config(cfg_path)
        self.cfg = cfg.system
        ckpt = torch.load(ckpt, map_location="cpu")
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material) if self.cfg.material_type != 'none' else None
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        ) if self.cfg.background_type != 'none' else None
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
        ckpt['state_dict']['background.env_color'] = torch.tensor([0.5, 0.5, 0.5])

        self.load_state_dict(ckpt['state_dict'], strict=True)
        self.geometry.encoding.encoding.disable_mask = True

    
    def forward(
            self,
            **kwargs
        ) -> Dict[str, Float[Tensor, "..."]]:
        # with torch.no_grad():
        out = self.renderer(**kwargs)
        comp_rgb = out["comp_rgb"]
        save_image(out['comp_rgb'].permute(0, 3, 1, 2), f'debug_data/magic123_gt.png')
    
        return {"comp_rgb": comp_rgb}