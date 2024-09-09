import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from threestudio.systems.utils import parse_optimizer

from torchvision.utils import save_image
import torch.nn.functional as F


@threestudio.register("interactive3d-system")
class Interactive3dSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

        self.init_cnt = 0
        if self.cfg.init:
            if self.cfg.init_type == "gsgen":
                self.init_renderer = threestudio.find('gs-renderer')(ckpt=self.cfg.init_ckpt)
            elif self.cfg.init_type == "magic123":
                self.init_renderer = threestudio.find('magic123-renderer')(ckpt=self.cfg.init_ckpt)
            elif self.cfg.init_type == "threestudio":
                self.init_renderer = threestudio.find('threestudio-renderer')(ckpt=self.cfg.init_ckpt)

        if self.cfg.edit_guidance_type != "":
            if self.cfg.edit_guidance_type != "copy-guidance":
                self.edit_guidance = threestudio.find(self.cfg.edit_guidance_type)(self.cfg.edit_guidance)
            else:
                self.edit_guidance = self.guidance
        else:
            self.edit_guidance = None
        self.edit_prompt_dict = dict()
        
        # save initial prompt
        self.edit_prompt_dict[self.prompt_processor.prompt] = self.prompt_utils

        if self.cfg.geometry.get('edit_pos_encoding_config') is not None:
            self.super_hash = True

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
            elif k.startswith("edit_guidance."):
                checkpoint['state_dict'].pop(k)
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        if self.edit_guidance is not None and self.cfg.edit_guidance_type == "copy-guidance":
            edit_guidance_state_dict = {"edit_guidance."+k : v for (k,v) in self.edit_guidance.state_dict().items()}
            checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict, **edit_guidance_state_dict}
        elif self.edit_guidance is not None and self.cfg.edit_guidance_type == "stable-diffusion-vsd-guidance":
            vsd_edit_state_dict = dict(self.edit_guidance.state_dict())
            vsd_edit_state_dict = {"edit_guidance."+k : v for (k,v) in vsd_edit_state_dict.items()}
            checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict, **vsd_edit_state_dict}
        else:
            checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        # load super hash
        # geometry.edit_encoding_list.3.encoding.encoding.params
        part_edit_flag = "geometry.edit_encoding_list.0.encoding.encoding.params" in list(checkpoint['state_dict'].keys())
        if self.cfg.geometry.get('edit_pos_encoding_config') is not None and "geometry.edit_encoding.encoding.encoding.params" not in list(checkpoint['state_dict'].keys()) and not part_edit_flag:
            edit_geometry_state_dict = {"geometry." + k: v for (k,v) in self.geometry.state_dict().items() if k.startswith('edit_')}
            checkpoint['state_dict'] = {**checkpoint['state_dict'], **edit_geometry_state_dict} # random initialize super hash
            not_load_optim = True

            if not_load_optim:
                # NOTE: save optim dict here
                current_optim = parse_optimizer(self.cfg.optimizer, self)
                checkpoint['optimizer_states'][0] = current_optim.state_dict()
            else:
                checkpoint['optimizer_states'][0]['state'] = {}
                edit_param_groups = []
                for param_dict in checkpoint['optimizer_states'][0]['param_groups']:
                    if param_dict['name'] == 'geometry.encoding':
                        edit_dict = {}
                        for k, v in param_dict.items():
                            if k == 'name':
                                edit_dict['name'] = 'geometry.edit_encoding'
                            else:
                                edit_dict[k] = v
                        edit_param_groups.append(edit_dict)
                    elif param_dict['name'] == 'geometry.density_network':
                        edit_dict = {}
                        for k, v in param_dict.items():
                            if k == 'name':
                                edit_dict['name'] = 'geometry.edit_density_network'
                            else:
                                edit_dict[k] = v
                        edit_param_groups.append(edit_dict) 
                    elif param_dict['name'] == 'geometry.feature_network':
                        edit_dict = {}
                        for k, v in param_dict.items():
                            if k == 'name':
                                edit_dict['name'] = 'geometry.edit_feature_network'
                            else:
                                edit_dict[k] = v
                        edit_param_groups.append(edit_dict)
                # vsd
                if self.cfg.edit_guidance_type == "stable-diffusion-vsd-guidance":
                    guidance_dict = {}
                    for k, v in edit_dict.items():
                        if k == 'name':
                            guidance_dict['name'] = 'guidance'
                        elif k == 'lr':
                            guidance_dict['lr'] = 0.0001
                        elif k == 'params':
                            continue
                        else:
                            guidance_dict[k] = v
                    edit_param_groups.append(guidance_dict) 
                checkpoint['optimizer_states'][0]['param_groups'].extend(edit_param_groups)

        return 

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
            elif k.startswith("edit_guidance."):
                checkpoint['state_dict'].pop(k)
        return 

    def forward(self, batch: Dict[str, Any], render_org=False, force_diffuse_light_color=None) -> Dict[str, Any]:
        return self.renderer(**batch, render_org=render_org, force_diffuse_light_color=force_diffuse_light_color)

    def training_step(self, batch, batch_idx):
        if 'edit_batch' in batch.keys():
            edit_batch = batch.pop('edit_batch')
            debug = False # TODO: refactor this
            if debug:
                print("======= set bg color to white =======")
                white_bg = torch.tensor([1., 1., 1.]).cuda()
                edit_batch['bg_color'] = white_bg
        else:
            edit_batch = None
        
        fix_view = batch.get('fix_view', False)
        if fix_view:
            print("======= set bg color to black =======")
            black_bg = torch.tensor([0., 0., 0.]).cuda()
            batch['bg_color'] = black_bg
        ###### NOTE: hard code to adjust only super frequency, not use now
        # if self.init_cnt % 2 == 1:
        #     self.cfg.only_super = False
        #     edit_batch = None
        # else:
        #     self.cfg.only_super = True
        ###### end hard code
        if not self.cfg.only_super:
            out = self(batch)
        else:
            out = {}

        if fix_view:
            save_imgs = out['comp_rgb'].permute(0, 3, 1, 2)
            prompt = self.prompt_processor.prompt
            os.makedirs(f'debug_data/fix_views/{prompt}', exist_ok=True)
            for i in range(save_imgs.shape[0]):
                save_image(save_imgs[i:i+1], 'debug_data/fix_views/{}/{:03d}.png'.format(prompt, i))
            print('=====SAVED AND EXIT!=====')
            exit()

        assert out != {} or edit_batch is not None
        if edit_batch is not None:
            edit_batch['super_hash'] = self.super_hash
            edit_out = self(edit_batch)

            save_imgs = edit_out['comp_rgb'].permute(0, 3, 1, 2)
            prompt = self.prompt_processor.prompt
            edit_prompt = edit_batch['edit_prompt']

            #NOTE: render org
            if self.cfg.render_org:
                edit_batch['bg_color'] = edit_out['bg_color']
                force_diffuse_light_color = self.material.cur_diffuse_light_color
                org_out = self(edit_batch, render_org=True, force_diffuse_light_color=force_diffuse_light_color)
                org_imgs = org_out['comp_rgb'].detach()
                org_opacity = org_out['opacity'].detach()
                edit_batch['org_imgs'] = org_imgs
                edit_batch['org_opacity'] = org_opacity
                save_imgs = torch.cat([save_imgs, org_imgs.permute(0, 3, 1, 2)], dim=0)

            os.makedirs(f'debug_data', exist_ok=True)
            save_image(save_imgs, f'debug_data/{prompt}_edit.png')
            
            if edit_prompt is not None:
                if self.edit_prompt_dict.get(edit_prompt, None) is None:
                    print(f"====== edit prompt is : {edit_prompt} =======")
                    prev_prompt = self.prompt_processor.prompt
                    self.prompt_processor.prompt = edit_prompt
                    edit_prompt_utils = self.prompt_processor()
                    self.prompt_processor.prompt = prev_prompt
                    self.edit_prompt_dict[edit_prompt] = edit_prompt_utils
        # init
        if self.cfg.init:
            if batch.get('sampled_cameras', None) is not None:
                self.init_cnt += 1
                if self.cfg.init_type == "threestudio":
                    self.background.cfg.random_aug = True
                    bg_color = self.background.cur_bg
                    force_shading = self.material.cur_shading
                    force_diffuse_light_colo = self.material.cur_diffuse_light_color
                    init_out = self.init_renderer(bg_color=bg_color, force_shading=force_shading, force_diffuse_light_color=force_diffuse_light_colo, **batch)
                else:
                    init_out = self.init_renderer(**batch)
                pred_rgb = out['comp_rgb']
                gt_rgb = init_out['comp_rgb']
                
                save_imgs = torch.cat([gt_rgb, pred_rgb], dim=0)
                save_imgs = save_imgs.permute(0, 3, 1, 2)
                if self.init_cnt % 10 == 0:
                    os.makedirs(f'debug_data', exist_ok=True)
                    save_image(save_imgs, 'debug_data/train_init_shape.png')
                loss = F.l1_loss(pred_rgb, gt_rgb)
                return loss
            else:
                print("======== Start SDS, enabel random bg ==========")
                self.cfg.init = False
                self.background.cfg.random_aug = True
        else:
            self.init_cnt += 1
            if self.init_cnt % 10 == 0 and 'comp_rgb' in out.keys():
                save_imgs = out['comp_rgb'].permute(0, 3, 1, 2)
                prompt = self.prompt_processor.prompt
                os.makedirs(f'debug_data', exist_ok=True)
                save_image(save_imgs, f'debug_data/{prompt}.png')

        if not self.cfg.only_super:
            guidance_out = self.guidance(
                out["comp_rgb"], self.prompt_utils, **batch
            )

        loss = 0.0

        if not self.cfg.only_super:
            for name, value in guidance_out.items():
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        # if edit_batch is not None and self.edit_guidance is not None and self.edit_prompt_utils is not None:
        if edit_batch is not None and self.edit_guidance is not None and self.edit_prompt_dict.get(edit_prompt, None) is not None:
            # compute edit guidance loss
            edit_prompt_utils = self.edit_prompt_dict[edit_prompt]
            edit_guidance_inp = edit_out["comp_rgb"]
            edit_guidance_out = self.edit_guidance(
                edit_guidance_inp, edit_prompt_utils, **edit_batch, rgb_as_latents=False,
            )
            # get loss
            for name, value in edit_guidance_out.items():
                self.log(f"train/edit_{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(self.cfg.edit_loss[name.replace("loss_", "lambda_")])
                    
            # get orient loss
            if self.C(self.cfg.loss.lambda_orient) > 0:
                if "normal" not in edit_out:
                    raise ValueError(
                        "Normal is required for orientation loss, no normal is found in the output."
                    )
                loss_orient = (
                    edit_out["weights"].detach()
                    * dot(edit_out["normal"], edit_out["t_dirs"]).clamp_min(0.0) ** 2
                ).sum() / (edit_out["opacity"] > 0).sum()
                self.log("train/loss_orient_edit", loss_orient)
                
                loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_orient) > 0 and not self.cfg.only_super:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0 and not self.cfg.only_super:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0 and not not self.cfg.only_super:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0 and not self.cfg.only_super:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if hasattr(self.cfg.loss, "lambda_eikonal") and self.C(self.cfg.loss.lambda_eikonal) > 0:
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        #NOTE: remove bg
        if self.cfg.eval_white_bg:
            white_bg = torch.tensor([1., 1., 1.]).cuda()
            # white_bg = torch.tensor([0., 0., 0.]).cuda()
            batch['bg_color'] = white_bg
    
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        #NOTE: remove bg
        if self.cfg.eval_white_bg:
            white_bg = torch.tensor([1., 1., 1.]).cuda()
            # white_bg = torch.tensor([0., 0., 0.]).cuda()
            batch['bg_color'] = white_bg
        
        out = self(batch)
        # NOTE: get sr video
        edit_guidance_inp = out["comp_rgb"]
        sr_img = self.edit_guidance(
                edit_guidance_inp, None, **batch, rgb_as_latents=False,
                only_return_sr=True
            )['sr']
        out["comp_rgb"] = sr_img
        
        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            ),
            # + (
            #     [
            #         {
            #             "type": "rgb",
            #             "img": out["comp_normal"][0],
            #             "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
            #         }
            #     ]
            #     if "comp_normal" in out
            #     else []
            # ), # normal
            # + [
            #     {
            #         "type": "grayscale",
            #         "img": out["opacity"][0, :, :, 0],
            #         "kwargs": {"cmap": None, "data_range": (0, 1)},
            #     },
            # ], # NOTE: not save opacity
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
