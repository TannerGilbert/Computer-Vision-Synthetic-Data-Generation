import json
import warnings
from pathlib import Path
import random
import base64
from io import BytesIO

from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage import measure
from shapely.geometry import Polygon
import albumentations as A


class SyntheticImageGenerator:

    def __init__(self, input_dir: str, output_dir: str, image_number: int, max_objects_per_image: int, image_width: int, image_height: int, occlude: bool, augmentation_path: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_number = image_number
        self.max_objects_per_image = max_objects_per_image
        self.image_width = image_width
        self.image_height = image_height
        self.occlude = occlude
        self.zero_padding = 8
        self.augmentation_path = Path(augmentation_path)

        self._validate_input_directory()
        self._validate_output_directory()
        self._validate_augmentation_path()

    def _validate_input_directory(self):
        # Check if directory exists
        assert self.input_dir.exists(), f'input_dir does not exist: {self.input_dir}'

        # Check if directory contains a foregrounds and backgrounds directory
        for p in self.input_dir.glob('*'):
            if p.name == 'foregrounds':
                self.foregrounds_dir = p
            elif p.name == 'backgrounds':
                self.backgrounds_dir = p

        assert self.foregrounds_dir is not None, 'foregrounds sub-directory was not found in the input_dir'
        assert self.backgrounds_dir is not None, 'backgrounds sub-directory was not found in the input_dir'

        self._validate_and_process_foregrounds()
        self._validate_and_process_backgrounds()

    def _validate_and_process_foregrounds(self):
        self.foregrounds_dict = dict()

        for category in self.foregrounds_dir.glob('*'):
            # check if we have a directory
            if not category.is_dir():
                warnings.warn(f'File found in foregrounds directory, ignoring: {category}')
                continue

            # Add images inside category folder to foregrounds dictionary
            self.foregrounds_dict[category.name] = list(category.glob('*.png'))

        assert len(self.foregrounds_dict) > 0, f'No valid foreground images were found in directory: {self.foregrounds_dir} '

    def _validate_and_process_backgrounds(self):
        self.background_images = []

        for ext in ('*.png', '*.jpg', '*jpeg'):
            self.background_images.extend(self.backgrounds_dir.glob(ext))

        assert len(self.background_images) > 0, f'No valid background images were found in directory: {self.backgrounds_dir}'

    def _validate_output_directory(self):
        # Check if directory is empty
        assert len(list(self.output_dir.glob('*'))) == 0, f'output_dir is not empty: {self.output_dir}'

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

    def _validate_augmentation_path(self):
        # Check if augmentation pipeline file exists
        if self.augmentation_path.is_file() and self.augmentation_path.suffix == 'yml':
            self.transforms = A.load(self.augmentation_path, data_format='yaml')
        else:
            self.transforms = None
            warnings.warn(f'{self.augmentation_path} is not a file. No augmentations will be applied')

    def _generate_image(self, image_number: int):
        # Randomly choose a background image
        background_image_path = random.choice(self.background_images)

        num_foreground_images = random.randint(1, self.max_objects_per_image)
        foregrounds = []
        for i in range(num_foreground_images):
            # Randomly choose a foreground
            category = random.choice(list(self.foregrounds_dict.keys()))
            foreground_path = random.choice(self.foregrounds_dict[category])

            foregrounds.append({
                'category': category,
                'image_path': foreground_path
            })

        # Compose foregrounds and background
        composite, annotations = self._compose_images(foregrounds, background_image_path)

        # Create the file name (used for both composite and mask)
        save_filename = f'{image_number:0{self.zero_padding}}'

        # Save image to output folder
        output_path = self.output_dir / f'{save_filename}.jpg'
        composite = composite.convert('RGB')
        composite.save(output_path)

        # Save annotations
        annotations['imagePath'] = f'{save_filename}.jpg'
        annotations_output_path = self.output_dir / f'{save_filename}.json'
        with open(annotations_output_path, 'w+') as output_file:
            json.dump(annotations, output_file)

    def _compose_images(self, foregrounds, background_image_path):
        # Open background image and convert to RGBA
        background_image = Image.open(background_image_path)
        background_image = background_image.convert('RGBA')

        # Resize background to desired width and height
        bg_width, bg_height = background_image.size

        if bg_width >= self.image_width and bg_height >= self.image_height:
            crop_x_pos = random.randint(0, bg_width - self.image_width)
            crop_y_pos = random.randint(0, bg_height - self.image_height)
            composite = background_image.crop(
                (crop_x_pos, crop_y_pos, crop_x_pos + self.image_width, crop_y_pos + self.image_height))
        else:
            composite = background_image.resize((self.image_width, self.image_height), Image.ANTIALIAS)

        annotations = dict()
        annotations['shapes'] = []
        annotations['imageWidth'] = self.image_width
        annotations['imageHeight'] = self.image_height

        for fg in foregrounds:
            fg_image = Image.open(fg['image_path'])
            # Perform transformations
            fg_image = self._transform_foreground(fg_image)

            # Choose a random x,y position
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, \
                f'foreground {fg["image_path"]} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({self.image_width}x{self.image_height}), check your input parameters'
            foreground_position = (random.randint(0, max_x_position), random.randint(0, max_y_position))

            # Create a new foreground image as large as the composite and paste it on top
            new_fg_image = Image.new('RGBA', composite.size, color=(0, 0, 0, 0))
            new_fg_image.paste(fg_image, foreground_position)

            # Extract the alpha channel from the foreground and paste it into a new image the size of the composite
            alpha_mask = fg_image.getchannel(3)
            new_alpha_mask = Image.new('L', composite.size, color=0)
            new_alpha_mask.paste(alpha_mask, foreground_position)
            composite = Image.composite(new_fg_image, composite, new_alpha_mask)

            # Grab the alpha pixels above a specified threshold
            alpha_threshold = 200
            mask_arr = np.array(np.greater(np.array(new_alpha_mask), alpha_threshold), dtype=np.uint8)
            mask = np.float32(mask_arr)  # This is composed of 1s and 0s
            contours = measure.find_contours(mask, 0.5, positive_orientation='low')

            annotation = dict()
            annotation['points'] = []
            annotation['label'] = fg['category']
            annotation['group_id'] = None
            annotation['shape_type'] = "polygon"
            annotation['flags'] = {}

            for contour in contours:
                poly = Polygon(contour)
                poly = poly.simplify(1.0, preserve_topology=False)

                if poly.area > 16:  # Ignore tiny polygons
                    if poly.geom_type == 'MultiPolygon':
                        # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
                        poly = poly.convex_hull

                    if poly.geom_type == 'Polygon':  # Ignore if still not a Polygon (could be a line or point)
                        segmentation = list(zip(*reversed(poly.exterior.coords.xy)))
                        annotation['points'].extend(segmentation)

            annotations['shapes'].append(annotation)

        # Save images as base64 string
        buffered = BytesIO()
        composite.convert('RGB').save(buffered, format="JPEG")
        annotations['imageData'] = base64.b64encode(buffered.getvalue()).decode()

        return composite, annotations

    def _transform_foreground(self, fg_image):
        if self.transforms:
            return Image.fromarray(self.transforms(image=np.array(fg_image))['image'])
        return fg_image

    def generate_images(self):
        for i in tqdm(range(1, self.image_number + 1)):
            self._generate_image(i)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Synthetic Image Generator')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the input directory. It must contain a backgrounds directory and a foregrounds directory')
    parser.add_argument('--output_dir', type=str, required=True, help='The directory where images and label files will be placed')
    parser.add_argument('--augmentation_path', type=str, default='transform.yml', help='Path to albumentations augmentation pipeline file')
    parser.add_argument('--image_number', type=int, required=True, help='Number of images to create')
    parser.add_argument('--max_objects_per_image', type=int, default=3, help='Maximum number of objects per images')
    parser.add_argument('--image_width', type=int, default=640, help='Width of the output images')
    parser.add_argument('--image_height', type=int, default=480, help='Height of the output images')
    parser.add_argument('--occlude', action='store_true', help="Whether or not the objects can overlap")

    args = parser.parse_args()

    data_generator = SyntheticImageGenerator(args.input_dir, args.output_dir, args.image_number, args.max_objects_per_image, args.image_width, args.image_height, args.occlude, args.augmentation_path)
    data_generator.generate_images()
