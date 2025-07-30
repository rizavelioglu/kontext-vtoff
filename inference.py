from loguru import logger
import os
import torch
import argparse
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images using FluxKontext pipeline')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder containing images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to output folder for processed images')
    parser.add_argument('--lora_folder', type=str, required=True, help='Path to folder where lora weights are stored')
    args = parser.parse_args()

    pipeline = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16).to('cuda')
    pipeline.load_lora_weights(args.lora_folder, weight_name='pytorch_lora_weights.safetensors')

    # Define input and output folders from arguments
    input_folder = args.input_folder
    output_folder = args.output_folder

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files from the input folder
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(image_extensions)]

    logger.info(f"Found {len(image_files)} images to process.")

    for i, image_file in enumerate(image_files):
        logger.info(f"Processing image {i+1}/{len(image_files)}: {image_file}")

        # Get filename without extension
        image_filename = os.path.splitext(image_file)[0]

        image = pipeline(
            image = load_image(os.path.join(input_folder, image_file)),
            prompt = 'extract only the upperbody garment over a white background, product photography style',
            guidance_scale = 2.5,
            generator=torch.Generator(device='cuda').manual_seed(42),
        ).images[0]

        # Save the output image
        output_path = os.path.join(output_folder, f'{image_filename}_out.png')
        image.save(output_path)

    logger.success("Processing complete!")


if __name__ == "__main__":
    main()
