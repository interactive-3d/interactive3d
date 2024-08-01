import torch
from diffusers import PixArtAlphaPipeline, ConsistencyDecoderVAE, AutoencoderKL
from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionPipeline

scheduler_params =  {
  "beta_end": 0.02, # 0.012,
  "beta_schedule": "linear", # "scaled_linear",
  "beta_start": 0.0001, # 0.00085,
  "dynamic_thresholding_ratio": 0.995,
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

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", torch_dtype=torch.float16, use_safetensors=True, scheduler=scheduler)

# Enable memory optimizations.
pipe.enable_model_cpu_offload()

prompt = "A gundam robot holding a sword with angel wings"
image = pipe(prompt, num_inference_steps=50).images[0]
image.save("./test_pixart.png")