import os
import glob
import json
from PIL import Image, ImageDraw
from io import BytesIO
import base64
import numpy as np
from shapely.geometry import Polygon


def extract_objects_from_labelme_data(input_dir, output_dir):
    # Create output directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    # Get path to all json files in the input directory
    labelme_json_paths = glob.glob(os.path.join(input_dir, "*.json"))

    label_counts = dict()

    for json_file in labelme_json_paths:
        # Open json file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Load base64 image
        im = Image.open(BytesIO(base64.b64decode(data['imageData']))).convert('RGBA')
        im_array = np.asarray(im)

        # Loop through all the annotations
        for annotation in data['shapes']:
            label = annotation['label']

            if label not in label_counts:
                label_counts[label] = 0
                os.makedirs(os.path.join(output_dir, label), exist_ok=True)

            # extract object from image
            # based on https://stackoverflow.com/questions/22588074/polygon-crop-clip-using-python-pil
            mask_im = Image.new('L', (im_array.shape[1], im_array.shape[0]), 0)
            ImageDraw.Draw(mask_im).polygon(tuple(map(tuple, annotation['points'])), outline=1, fill=1)
            mask = np.array(mask_im)

            # assemble new image (uint8: 0-255)
            new_img_array = np.empty(im_array.shape, dtype='uint8')

            # colors (three first columns, RGB)
            new_img_array[:, :, :3] = im_array[:, :, :3]

            # transparency (4th column)
            new_img_array[:, :, 3] = mask * 255

            # convert to image, crop and save
            new_im = Image.fromarray(new_img_array, "RGBA")
            x_min, y_min, x_max, y_max = Polygon(annotation['points']).bounds
            new_im = new_im.crop((x_min, y_min, x_max, y_max))
            new_im.save(os.path.join(output_dir, label, f'{label_counts[label]}.png'))
            label_counts[label] += 1


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract objects from data labeled with labelme')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to input images and labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Path where output images will be saved')
    args = parser.parse_args()

    extract_objects_from_labelme_data(args.input_dir, args.output_dir)
