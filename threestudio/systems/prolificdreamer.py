import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

from torchvision.utils import save_image


@threestudio.register("prolificdreamer-system")
class ProlificDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        # in ['coarse', 'geometry', 'texture']
        stage: str = "coarse"
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        if self.cfg.guidance_type == "multiview-diffusion-guidance":
            self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        self.cnt = 0
        # edit
        if self.cfg.edit_guidance_type != "":
            if self.cfg.edit_guidance_type != "copy-guidance":
                self.edit_guidance = threestudio.find(self.cfg.edit_guidance_type)(self.cfg.edit_guidance)
            else:
                self.edit_guidance = self.guidance
        else:
            self.edit_guidance = None
        self.edit_prompt_dict = dict()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.cfg.stage == "geometry":
            render_out = self.renderer(**batch, render_rgb=False)
        else:
            render_out = self.renderer(**batch)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()

    def training_step(self, batch, batch_idx):
        self.cnt += 1
        if 'edit_batch' in batch:
            edit_batch = batch.pop('edit_batch')
        else:
            edit_batch = None
        out = self(batch)

        if edit_batch is not None:
            edit_out = self(edit_batch)
            save_imgs = edit_out['comp_rgb'].permute(0, 3, 1, 2)
            prompt = self.prompt_processor.prompt
            edit_prompt = edit_batch['edit_prompt']
            save_image(save_imgs, f'debug_data/{prompt}_edit.png')
            # if edit_prompt is None:
            #     save_image(save_imgs, f'debug_data/{prompt}_edit.png')
            # else:
            #     if self.init_cnt % 10 == 0:
            #         save_image(save_imgs, f'debug_data/{prompt}_edit.png')
            # if self.edit_prompt_utils is None and edit_prompt is not None:
            if edit_prompt is not None:
                if self.edit_prompt_dict.get(edit_prompt, None) is None:
                    print(f"====== edit prompt is : {edit_prompt} =======")
                    prev_prompt = self.prompt_processor.prompt
                    self.prompt_processor.prompt = edit_prompt
                    edit_prompt_utils = self.prompt_processor()
                    self.prompt_processor.prompt = prev_prompt
                    self.edit_prompt_dict[edit_prompt] = edit_prompt_utils

        if self.cfg.stage == "geometry":
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
            )
        else:
            if self.cnt % 10 == 0 and 'comp_rgb' in out.keys():
                save_imgs = out['comp_rgb'].permute(0, 3, 1, 2)
                prompt = self.prompt_processor.prompt
                save_image(save_imgs, f'debug_data/{prompt}.png')
            guidance_inp = out["comp_rgb"]
            guidance_out = self.guidance(
                guidance_inp, self.prompt_utils, **batch, rgb_as_latents=False
            )

        loss = 0.0

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

        if self.cfg.stage == "coarse":
            if self.C(self.cfg.loss.lambda_orient) > 0:
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

            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

            # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
            # helps reduce floaters and produce solid geometry
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)
        elif self.cfg.stage == "geometry":
            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )

            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
        elif self.cfg.stage == "texture":
            pass
        else:
            raise ValueError(f"Unknown stage {self.cfg.stage}")

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
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

        if self.cfg.visualize_samples:
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][0]}-sample.png",
                [
                    {
                        "type": "rgb",
                        "img": self.guidance.sample(
                            self.prompt_utils, **batch, seed=self.global_step
                        )[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                    {
                        "type": "rgb",
                        "img": self.guidance.sample_lora(self.prompt_utils, **batch)[0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ],
                name="validation_step_samples",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        out = self(batch)
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
