from diffusers import StableDiffusionPipeline
import torch
from deep_translator import GoogleTranslator
import gradio as gr
import os
import sys
import torch
import open_clip
from PIL import Image
from open_clip import SimpleTokenizer
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel


def get_by_prompt(height=800, width=800):
    # Перевод на английский для нормального взаимодействия с моделями
    translated_prompt = GoogleTranslator(source='auto', target='english').translate(txt.value)
    images = pipe(
        prompt=translated_prompt,
        height=height,
        width=width,
        num_inference_steps=100,
        guidance_scale=0.5,
        num_images_per_prompt=1
    ).images
    return images[0]


with gr.Blocks() as demo:
    txt = gr.Textbox(label="Описание вашего будущего превью", lines=2)
    btn = gr.Button(value="Сгенерировать превью")
    out_img = gr.Image()
    btn.click(get_by_prompt, inputs=[], outputs=[out_img])

if __name__ == "__main__":
    sys.path.append('./images_mixing')

    # Подгрузка модельки dreamlike
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    # Загрузка моделей clip и coca
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )
    clip_model = CLIPModel.from_pretrained(
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16
    )
    coca_model = open_clip.create_model('coca_ViT-L-14', pretrained='laion2B-s13B-b90k').to('cuda')
    coca_model.dtype = torch.float16
    coca_transform = open_clip.image_transform(
        coca_model.visual.image_size,
        is_train=False,
        mean=getattr(coca_model.visual, 'image_mean', None),
        std=getattr(coca_model.visual, 'image_std', None),
    )
    coca_tokenizer = SimpleTokenizer()

    demo.launch()
