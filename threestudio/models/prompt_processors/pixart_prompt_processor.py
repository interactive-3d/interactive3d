import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import IFPipeline
from transformers import T5EncoderModel, T5Tokenizer

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from threestudio.utils.misc import cleanup
from threestudio.utils.typing import *

from threestudio.utils.misc import barrier, get_rank

from diffusers import PixArtAlphaPipeline

@dataclass
class DirectionConfig:
    name: str
    prompt: Callable[[str], str]
    negative_prompt: Callable[[str], str]
    condition: Callable[
        [Float[Tensor, "B"], Float[Tensor, "B"], Float[Tensor, "B"]],
        Float[Tensor, "B"],
    ]

@dataclass
class PromptProcessorOutput:
    text_embeddings: Float[Tensor, "N Nf"]
    text_embeddings_masks: Float[Tensor, "N Nf"]
    uncond_text_embeddings: Float[Tensor, "N Nf"]
    uncond_text_embeddings_masks: Float[Tensor, "N Nf"]
    text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    text_embeddings_vd_masks: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd: Float[Tensor, "Nv N Nf"]
    uncond_text_embeddings_vd_masks: Float[Tensor, "Nv N Nf"]
    directions: List[DirectionConfig]
    direction2idx: Dict[str, int]
    use_perp_neg: bool
    perp_neg_f_sb: Tuple[float, float, float]
    perp_neg_f_fsb: Tuple[float, float, float]
    perp_neg_f_fs: Tuple[float, float, float]
    perp_neg_f_sf: Tuple[float, float, float]

    def get_text_embeddings(
        self,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        view_dependent_prompting: bool = True,
    ) -> Float[Tensor, "BB N Nf"]:
        batch_size = elevation.shape[0]

        if view_dependent_prompting:
            # Get direction
            direction_idx = torch.zeros_like(elevation, dtype=torch.long)
            for d in self.directions:
                direction_idx[
                    d.condition(elevation, azimuth, camera_distances)
                ] = self.direction2idx[d.name]

            # Get text embeddings
            text_embeddings = self.text_embeddings_vd[direction_idx]  # type: ignore
            text_embeddings_masks = self.text_embeddings_vd_masks[direction_idx]  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings_vd[direction_idx]  # type: ignore
            uncond_text_embeddings_masks = self.uncond_text_embeddings_vd_masks[direction_idx]  # type: ignore
        else:
            text_embeddings = self.text_embeddings.expand(batch_size, -1, -1)  # type: ignore
            text_embeddings_masks = self.text_embeddings_masks.expand(batch_size, -1, -1)  # type: ignore
            uncond_text_embeddings = self.uncond_text_embeddings.expand(  # type: ignore
                batch_size, -1, -1
            )
            uncond_text_embeddings_masks = self.uncond_text_embeddings_masks.expand(  # type: ignore
                batch_size, -1, -1
            )

        # IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        return torch.cat([text_embeddings, uncond_text_embeddings], dim=0), torch.cat([text_embeddings_masks, uncond_text_embeddings_masks], dim=0)

    
MASK_DIR = 'path_to_your_mask_cache'
@threestudio.register("pixart-prompt-processor")
class PixartPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pretrained_model_name_or_path: str = "PixArt-alpha/PixArt-XL-2-1024-MS"

    cfg: Config

    ### these functions are unused, kept for debugging ###
    def configure_text_encoder(self) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = T5EncoderModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="text_encoder",
            load_in_8bit=True,
            variant="8bit",
            device_map="auto",
        )  # FIXME: behavior of auto device map in multi-GPU training
        
        self.pipe = PixArtAlphaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-XL-2-1024-MS",
            text_encoder=self.text_encoder,
            transformer=None,
            device_map="auto",
            local_files_only=True
        )

    def destroy_text_encoder(self) -> None:
        del self.text_encoder
        del self.pipe
        cleanup()

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Float[Tensor, "B 120 4096"], Float[Tensor, "B 120 4096"]]:
        # text_embeddings, uncond_text_embeddings
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipe.encode_prompt(
            prompt=prompt, do_classifier_free_guidance=True,
            negative_prompt=negative_prompt, 
            num_images_per_prompt=1,
            device=self.device,
            # clean_caption=True,
            max_sequence_length=120,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        return prompt_embeds, prompt_embeds

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        print("=========pretrained model name or path===========:", pretrained_model_name_or_path)
        max_length = 120 # 77
        tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        text_encoder = T5EncoderModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            load_in_8bit=True,
            device_map="auto",
            
        )
        with torch.no_grad():
            text_inputs = tokenizer(
                prompts,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids
            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                print(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )
            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.cuda()

            prompt_embeds = text_encoder(text_input_ids.cuda(), attention_mask=prompt_attention_mask)
            prompt_embeds = prompt_embeds[0]

            dtype = text_encoder.dtype
            device = prompt_attention_mask.device
            prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            num_images_per_prompt = 1
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
            prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
            
        for prompt, embedding in zip(prompts, prompt_embeds):
            torch.save(
                embedding,
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )
        
        for prompt, mask in zip(prompts, prompt_attention_mask):
            torch.save(
                mask,
                os.path.join(
                    MASK_DIR,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.pt",
                ),
            )

        del text_encoder
    
    def load_text_embeddings(self):
        # synchronize, to ensure the text embeddings have been computed and saved to cache
        barrier()
        self.text_embeddings = self.load_from_cache(self.prompt)[None, ...]
        self.uncond_text_embeddings = self.load_from_cache(self.negative_prompt)[
            None, ...
        ]
        self.text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd = torch.stack(
            [self.load_from_cache(prompt) for prompt in self.negative_prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text embeddings.")

        # load mask
        self.text_embeddings_masks = self.load_from_cache_mask(self.prompt)[None, ...]
        self.uncond_text_embeddings_masks = self.load_from_cache_mask(self.negative_prompt)[
            None, ...
        ]
        self.text_embeddings_vd_masks = torch.stack(
            [self.load_from_cache_mask(prompt) for prompt in self.prompts_vd], dim=0
        )
        self.uncond_text_embeddings_vd_masks = torch.stack(
            [self.load_from_cache_mask(prompt) for prompt in self.negative_prompts_vd], dim=0
        )
        threestudio.debug(f"Loaded text masks.")

    def load_from_cache(self, prompt):
        cache_path = os.path.join(
            self._cache_dir,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Text embedding file {cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(cache_path, map_location=self.device)
    
    def load_from_cache_mask(self, prompt):
        mask_cache_path = os.path.join(
            MASK_DIR,
            f"{hash_prompt(self.cfg.pretrained_model_name_or_path, prompt)}.pt",
        )
        if not os.path.exists(mask_cache_path):
            raise FileNotFoundError(
                f"Mask file {mask_cache_path} for model {self.cfg.pretrained_model_name_or_path} and prompt [{prompt}] not found."
            )
        return torch.load(mask_cache_path, map_location=self.device)

    def __call__(self) -> PromptProcessorOutput:
        return PromptProcessorOutput(
            text_embeddings=self.text_embeddings,
            text_embeddings_masks=self.text_embeddings_masks,
            uncond_text_embeddings=self.uncond_text_embeddings,
            uncond_text_embeddings_masks=self.uncond_text_embeddings_masks,
            text_embeddings_vd=self.text_embeddings_vd,
            text_embeddings_vd_masks=self.text_embeddings_vd_masks,
            uncond_text_embeddings_vd=self.uncond_text_embeddings_vd,
            uncond_text_embeddings_vd_masks=self.uncond_text_embeddings_vd_masks,
            directions=self.directions,
            direction2idx=self.direction2idx,
            use_perp_neg=self.cfg.use_perp_neg,
            perp_neg_f_sb=self.cfg.perp_neg_f_sb,
            perp_neg_f_fsb=self.cfg.perp_neg_f_fsb,
            perp_neg_f_fs=self.cfg.perp_neg_f_fs,
            perp_neg_f_sf=self.cfg.perp_neg_f_sf,
        )

