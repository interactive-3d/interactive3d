from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers import DDIMScheduler
import torch

scheduler_params =  {
  "beta_end": 0.012,
  "beta_schedule": "scaled_linear",
  "beta_start": 0.00085,
  # "dynamic_thresholding_ratio": 0.995,
  "clip_sample": False,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon", # "v_prediction",
  "timestep_spacing": "linspace",
  "set_alpha_to_one": False,
#   "skip_prk_steps": True,
#   "steps_offset": 1,
#   "trained_betas": None
}
scheduler = DDIMScheduler(**scheduler_params)

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, scheduler=scheduler
).to("cuda")

prompt = "A gundam robot holding a sword with angel wings, detailed, 8k"
image = pipeline(prompt=prompt).images[0]
image.save("./test_sdxl.png")