import torch
import torch.nn as nn
import sys
import threestudio
from threestudio.utils.typing import *
from threestudio.utils.config import ExperimentConfig, load_config
from torchvision.utils import save_image

@threestudio.register("threestudio-renderer")
class ThreestudioRenderer(nn.Module):
    def __init__(self, ckpt=None):
        super(ThreestudioRenderer, self).__init__()
        ckpt_root = 'path_to_your_ckpt_root'
        ckpt = ckpt_root + '/ckpts/last.ckpt'
        ckpt = torch.load(ckpt, map_location="cpu")

        cfg_path = ckpt_root + '/configs/parsed.yaml'
        cfg = load_config(cfg_path)
        self.cfg = cfg.system

        self.configure()
    
        self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"==== loaded state dict from {ckpt_root} ====")

    
    def configure(self) -> None:
        
        self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)

        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(
            self.cfg.background
        )
        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )
    
    def forward(
            self,
            bg_color=None,
            force_shading=None,
            **kwargs
        ) -> Dict[str, Float[Tensor, "..."]]:
        with torch.cuda.amp.autocast(enabled=False):
            out = self.renderer(bg_color=bg_color, force_shading=force_shading, **kwargs)
        comp_rgb = out["comp_rgb"]
        save_image(out['comp_rgb'].permute(0, 3, 1, 2), f'debug_data/three_gt.png')
    
        return {"comp_rgb": comp_rgb}