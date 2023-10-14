from diffusers import StableDiffusionPipeline
import torch
import tensorflow as tf
from deep_translator import GoogleTranslator
from matplotlib import pyplot as plt
from matplotlib import image as mpimg


def get_by_prompt(prompt, height=800, width=800):
    images = pipe(
        prompt=prompt,
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
    user_prompt = "мой временный промпт"

    # Перевод на английский для нормального взаимодействия с моделями
    translated_prompt = GoogleTranslator(source='auto', target='ru').translate(user_prompt)
    ready_images = get_by_prompt(translated_prompt)
    plt.imshow(ready_images[0])
    plt.show()
