import torch
import torch.nn as nn
import sys
import threestudio
from threestudio.utils.typing import *
from gs.gaussian_splatting import GaussianSplattingRenderer

import numpy as np

from torchvision.utils import save_image
import pdb
from omegaconf import OmegaConf

@threestudio.register("gs-renderer")
class GSRenderer(nn.Module):
    def __init__(self, ckpt=None):
        super(GSRenderer, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if ckpt is None:
            ckpt = "path_to_your_ckpt.pt"
        ckpt = torch.load(ckpt, map_location="cpu")
        cfg = OmegaConf.create(ckpt["cfg"])
        self.renderer = GaussianSplattingRenderer.load(
            cfg.renderer, ckpt["params"]
        ).to(device)
        cfg.type = 'fixed'
        cfg.color = [0.5, 0.5, 0.5]
        cfg.random_aug = False
        cfg.random_aug_prob = 0.
        self.renderer.setup_bg(cfg)
        print("===== [NOTE]: set gs bg color to 0.5 ====== ")
        self.cfg = cfg
    
    def forward(
            self,
            sampled_cameras,
            cam_id=None,
            gt_img=None,
            **kwargs
        ) -> Dict[str, Float[Tensor, "..."]]:
        with torch.cuda.amp.autocast(enabled=False):
            # rotate
            c2w = sampled_cameras['c2w']
            c2w = torch.cat(
                [c2w, torch.zeros_like(c2w[:, :1])], dim=1
            )
            c2w[:, 3, 3] = 1.0
            trans = torch.zeros_like(c2w).type(c2w.dtype) # b, 4, 4
            trans[:, 3, 3] = 1.0

            # trans[:, 0, 0] = -1. 
            # trans[:, 1, 1] = -1. 
            # trans[:, 2, 2] = 1.

            trans[:, 0, 0] = 0. 
            trans[:, 0, 1] = -1. 
            trans[:, 1, 0] = 1. 
            trans[:, 1, 1] = 0. 
            trans[:, 2, 2] = 1.

            c2w = torch.matmul(trans, c2w)
            sampled_cameras['c2w'] = c2w[:, :3].cuda()
            out = self.renderer(sampled_cameras, self.cfg.use_bg, self.cfg.rgb_only)
        comp_rgb = out["rgb"]
        # save_image(out['rgb'].permute(0, 3, 1, 2), f'debug_data/gsgen_gt.png')
    
        return {"comp_rgb": comp_rgb}