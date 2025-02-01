import os
from pdf2image import convert_from_path

def pdf_to_images(pdf_path, output_folder, dpi=300):
    """
    Converts a PDF to images and ensures the output folder exists.
    """
    if not os.path.exists(output_folder):  # Create folder if missing
        os.makedirs(output_folder)

    images = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    for i, img in enumerate(images):
        img_path = os.path.join(output_folder, f"page_{i+1}.png")
        img.save(img_path, "PNG")
        image_paths.append(img_path)

    return image_paths