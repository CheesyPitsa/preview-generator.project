from diffusers import StableDiffusionPipeline
import torch
from deep_translator import GoogleTranslator
import gradio as gr
import sys
import torch
import open_clip
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


def get_by_images(image, image_style):
    generator = torch.Generator(device="cuda").manual_seed(17)

    content_image = image
    style_image = image_style

    pipe_images = mixing_pipeline(
        num_inference_steps=50,
        content_image=content_image,
        style_image=style_image,
        content_prompt=None,
        style_prompt=None,
        noise_strength=0.4,
        slerp_latent_style_strength=0.25,
        slerp_prompt_style_strength=0.99,
        slerp_clip_image_style_strength=0.9,
        guidance_scale=9.0,
        batch_size=1,
        clip_guidance_scale=100,
        generator=generator,
        print_promts=True,
    ).images
    return pipe_images


with gr.Blocks() as demo:
    txt = gr.Textbox(label="Описание вашего будущего превью", lines=2)
    inpImg = gr.Image(type="pil")
    inpStyle = gr.Image(type="pil")
    btn = gr.Button(value="Сгенерировать превью по тексту")
    btnImg = gr.Button(value="Сгенерировать превью по изображениям")
    out_img = gr.Image()
    btn.click(get_by_prompt, inputs=[], outputs=[out_img])
    btnImg.click(get_by_images, inputs=[inpImg, inpStyle], outputs=[out_img])

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

    mixing_pipeline = DiffusionPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-2-base",
        "CompVis/stable-diffusion-v1-4",
        custom_pipeline="./images_mixing/images_mixing.py",
        clip_model=clip_model,
        feature_extractor=feature_extractor,
        coca_model=coca_model,
        coca_tokenizer=coca_tokenizer,
        coca_transform=coca_transform,
        torch_dtype=torch.float16,
    )
    mixing_pipeline = mixing_pipeline.to("cuda")

    demo.launch()
