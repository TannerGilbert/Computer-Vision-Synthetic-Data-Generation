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
import multiprocessing
from joblib import Parallel, delayed
from typing import List


class SyntheticImageGenerator:

    def __init__(self, input_dir: str, output_dir: str, image_number: int, max_objects_per_image: int, image_width: int, image_height: int, augmentation_path: str, scale_foreground_by_background_size: bool, scaling_factors: List[int], avoid_collisions: bool, parallelize: bool):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.image_number = image_number
        self.max_objects_per_image = max_objects_per_image
        self.image_width = image_width
        self.image_height = image_height
        self.zero_padding = 8
        self.augmentation_path = Path(augmentation_path)
        self.scale_foreground_by_background_size = scale_foreground_by_background_size
        self.scaling_factors = scaling_factors
        self.avoid_collisions = avoid_collisions
        self.parallelize = parallelize

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
        if self.augmentation_path.is_file() and self.augmentation_path.suffix == '.yml':
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

        fg_list = []

        for fg in foregrounds:
            fg_image = Image.open(fg['image_path'])

            # Resize foreground (based on https://github.com/basedrhys/cocosynth/commit/f0b5d4d97009a3a070ba9967ff536c7dd71af6ef)
            if not self.scale_foreground_by_background_size:
                # Apply random scale
                scale = random.random() * .5 + .5  # Pick something between .5 and 1
                new_size = (int(fg_image.size[0] * scale), int(fg_image.size[1] * scale))
            else:
                # Scale the foreground based on the size of the resulting image
                min_bg_len = min(self.image_width, self.image_height)

                rand_length = random.random() * (self.scaling_factors[1] - self.scaling_factors[0]) + self.scaling_factors[0]
                long_side_len = rand_length * min_bg_len
                # Scale the longest side of the fg to be between the random length % of the bg
                if fg_image.size[0] > fg_image.size[1]:
                    long_side_dif = long_side_len / fg_image.size[0]
                    short_side_len = long_side_dif * fg_image.size[1]
                    new_size = (int(long_side_len), int(short_side_len))
                else:
                    long_side_dif = long_side_len / fg_image.size[1]
                    short_side_len = long_side_dif * fg_image.size[0]
                    new_size = (int(short_side_len), int(long_side_len))

            fg_image = fg_image.resize(new_size, resample=Image.BICUBIC)

            # Perform transformations
            fg_image = self._transform_foreground(fg_image)

            # Choose a random x,y position
            max_x_position = composite.size[0] - fg_image.size[0]
            max_y_position = composite.size[1] - fg_image.size[1]
            assert max_x_position >= 0 and max_y_position >= 0, \
                f'foreground {fg["image_path"]} is too big ({fg_image.size[0]}x{fg_image.size[1]}) for the requested output size ({self.image_width}x{self.image_height}), check your input parameters'
            foreground_position = (random.randint(0, max_x_position), random.randint(0, max_y_position))

            # avoid collisions of foreground objects (based on https://github.com/basedrhys/cocosynth/commit/d009a0de17b154ca3b469e8d4c0a7afa8fa51271)
            if self.avoid_collisions:
                fg_rect = [foreground_position[0],  # x1
                           foreground_position[1],  # y1
                           foreground_position[0] + fg_image.size[0],  # x2
                           foreground_position[1] + fg_image.size[1]]  # y2

                visited_centroids = []
                colliding_point = self._is_colliding(fg_rect, fg_list)

                while colliding_point is not None:
                    # Move the fg away from the colliding point
                    step_size = 50
                    curr_centroid_x = int((fg_rect[0] + fg_rect[2]) / 2)
                    curr_centroid_y = int((fg_rect[1] + fg_rect[3]) / 2)
                    new_centroid_pos = self._get_new_centroid_pos(colliding_point,
                                                                  (curr_centroid_x, curr_centroid_y),
                                                                  step_size)

                    if self._visited_point_before(new_centroid_pos, visited_centroids):
                        print("Tried to re-visit point {}".format(new_centroid_pos))
                        fg_rect = None
                        break
                    visited_centroids.append(new_centroid_pos)

                    fg_rect = self._get_rect_position(new_centroid_pos, fg_image)
                    colliding_point = self._is_colliding(fg_rect, fg_list)

                if fg_rect is None or self._outside_img(composite, fg_rect):
                    # print("Outside image {}".format(fg_rect))
                    continue
                else:
                    paste_position = (int(fg_rect[0]), int(fg_rect[1]))
                    fg_list.append(fg_rect)

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

    @staticmethod
    def _get_point_to_move_from(colliding_centroids):
        """
        Average all the centroid locations that the fg was colliding with
        This gives a point for the fg to move away from
        input: array of centroids -> [(200, 100), (50, 20), ...]
        """
        return np.mean(colliding_centroids, 0)

    def _is_colliding(self, fg_rect, fg_rect_list):
        """
        Check if the current foreground object is colliding with any foregrounds
        we've placed on the image already. The overlap threshold controls how much overlap
        of the rectangles is allowed. The overlap is a proportion of the total area of this
        foreground object, as opposed to an absolute value
        """
        overlap_thresh = 0.3

        colliding_centroids = []

        for rect in fg_rect_list:
            x1, y1, x2, y2 = fg_rect
            comp_x1, comp_y1, comp_x2, comp_y2 = rect
            x_overlap = max(0, min(comp_x2, x2) - max(comp_x1, x1))
            y_overlap = max(0, min(comp_y2, y2) - max(comp_y1, y1))

            rect_area = (x2 - x1) * (y2 - y1)
            total_overlap_area = x_overlap * y_overlap

            prop_overlap = total_overlap_area / float(rect_area)

            if prop_overlap > overlap_thresh:
                colliding_centroid = (comp_x2 - comp_x1, comp_y2 - comp_y1)
                colliding_centroids.append(colliding_centroid)

        if len(colliding_centroids) == 0:
            return None
        else:
            return self._get_point_to_move_from(colliding_centroids)

    @staticmethod
    def _get_new_centroid_pos(pt_a, pt_b, step_size):
        """
        https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
        Given the point to move from (pt_a), and our current centroid (pt_b),
        move further away along the line i.e.:
        a -------- b --<step_size>-- new_point
        """
        v = np.subtract(pt_b, pt_a)
        norm_v = v / np.linalg.norm(v)

        new_point = tuple(map(int, pt_b + (step_size * norm_v)))  # Convert to an int tuple
        # print("Moving from {} to {}".format(pt_b, new_point))
        return new_point

    @staticmethod
    def _get_rect_position(centroid, fg_image):
        """
        Convert a centroid to the corresponding rectangle, given a foreground image
        """
        width, height = fg_image.size
        x1 = centroid[0] - (width / 2)
        y1 = centroid[1] - (height / 2)
        x2 = centroid[0] + (width / 2)
        y2 = centroid[1] + (height / 2)
        return [x1, y1, x2, y2]

    @staticmethod
    def _visited_point_before(coll_pt, coll_pts):
        """
        Check if we've visited this point before.
        Stops collision avoidance from getting stuck in between a group of points
        """
        for pt in coll_pts:
            if coll_pt[0] == pt[0] and coll_pt[1] == pt[1]:
                return True

        return False

    @staticmethod
    def _outside_img(img, fg_rect):
        """
        Don't paste the foreground if the centroid is outside the image
        """
        curr_centroid_x = int((fg_rect[0] + fg_rect[2]) / 2)
        curr_centroid_y = int((fg_rect[1] + fg_rect[3]) / 2)

        return (curr_centroid_x < 0 or
                curr_centroid_x > img.size[0] or
                curr_centroid_y < 0 or
                curr_centroid_y > img.size[1])

    def _transform_foreground(self, fg_image):
        if self.transforms:
            return Image.fromarray(self.transforms(image=np.array(fg_image))['image'])
        return fg_image

    def generate_images(self):
        if self.parallelize:
            Parallel(n_jobs=multiprocessing.cpu_count())(delayed(self._generate_image)(i) for i in tqdm(range(1, self.image_number + 1)))
        else:
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
    parser.add_argument('--scale_foreground_by_background_size', default=True, action='store_false', help='Whether the foreground images should be scaled based on the background size (default=true)')
    parser.add_argument('--scaling_factors', type=float, nargs=2, default=(0.25, 0.5), help='Min and Max percentage size of the short side of the background image')
    parser.add_argument('--avoid_collisions', default=True, action='store_false', help='Whether or not to avoid collisions (default=true)')
    parser.add_argument('--parallelize', default=False, action='store_true', help='Whether or not to use multiple cores (default=false)')

    args = parser.parse_args()

    data_generator = SyntheticImageGenerator(args.input_dir, args.output_dir, args.image_number, args.max_objects_per_image, args.image_width, args.image_height, args.augmentation_path, args.scale_foreground_by_background_size, args.scaling_factors, args.avoid_collisions, args.parallelize)
    data_generator.generate_images()
