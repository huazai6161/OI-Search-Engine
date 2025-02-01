import os
from PdfToImage import pdf_to_images
from ImageToText import images_to_text

def PDFconvertor(folder):
    pdf_path = os.path.join(folder, "solution-pdf")
    img_path = os.path.join(folder, "solution-img")
    md_path = os.path.join(folder, "solution")
    
    if(not os.path.exists(img_path)):
        os.makedirs(img_path)
    if(not os.path.exists(md_path)):
        os.makedirs(md_path)

    for pdf in os.listdir(pdf_path):
        task_name = os.path.splitext(pdf)[0]
        img_folder = os.path.join(img_path, task_name)
        path = os.path.join(pdf_path, pdf)
        pdf_to_images(path, img_folder)

        output_file = os.path.join(md_path, task_name + ".md")
        images_to_text(img_folder, output_file, prompt="Extract the equations and text from this solution to the markdown format, and add '$' on both sides of math terms. Ignore the comments and other unrelated UI, focusing on the solution. Your response could only contain the translation of the solution.")

if __name__ == "__main__":
    PDFconvertor("data/questions/Luogu")