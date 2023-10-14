from diffusers import StableDiffusionPipeline
import torch


def get_by_prompt(prompt, height, width):
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=100,
        guidance_scale=0.5,
        num_images_per_prompt=1
    ).images


if __name__ == '__main__':
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
