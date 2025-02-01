import openai
import base64
import os
from config import OPENAI_API_KEY, COMPLETION_MODEL

def encode_image(image_path):
    """Encodes an image to base64 for API submission."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def send_images_to_gpt(image_paths, prompt):
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    images_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image(img_path)}"},
        }
        for img_path in image_paths
    ]

    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[
            {"role": "system", "content": f"You are an expert in informatics olympiad."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + images_content
            }
        ]
    )

    return response.choices[0].message.content

def images_to_text(image_folder, output_file, prompt):

    with open(output_file, "w", encoding="utf-8") as f:
        image_paths = [os.path.join(image_folder, img_file) for img_file in sorted(os.listdir(image_folder))]
        f.write(send_images_to_gpt(image_paths, prompt))

    print(f"All translations saved to {output_file}")
