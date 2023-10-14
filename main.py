from diffusers import StableDiffusionPipeline
import torch
from deep_translator import GoogleTranslator
import gradio as gr
import os


def get_by_prompt(prompt, height=800, width=800):
    # Перевод на английский для нормального взаимодействия с моделями
    translated_prompt = GoogleTranslator(source='auto', target='ru').translate(prompt)
    images = pipe(
        prompt=translated_prompt,
        height=height,
        width=width,
        num_inference_steps=100,
        guidance_scale=0.5,
        num_images_per_prompt=1
    ).images
    return images


if __name__ == '__main__':
    # Подгрузка модельки dreamlike (надо скачать)
    model_id = "dreamlike-art/dreamlike-diffusion-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    # Здесь будет получение данных с формы

    with gr.Blocks() as demo:
        txt = gr.Textbox(label="Описание вашего будущего превью", lines=2)
        btn = gr.Button(value="Сгенерировать превью")
        out_img = gr.Image(label="Результат")
        btn.click(get_by_prompt(txt)[0], inputs=[txt], outputs=[out_img])

        with gr.Row():
            im = gr.Image()
            im_2 = gr.Image()
