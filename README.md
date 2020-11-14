# Synthetic Data Generation for Object Detection and Instance Segmentation

![Output Example](doc/images/output_example.PNG)

This code allows you to create a synthetic data-set, for Instance Segmentation or Object Detection. The [data_generation.py](data_generation.py) script outputs data in [LabelMe format](https://roboflow.com/formats/labelme-json), which can also be converted to other formats like the [COCO JSON format](https://cocodataset.org/).

 ## Installation
 
 1. Clone the repository 
    ```git clone https://github.com/TannerGilbert/Object-Detection-Synthetic-Data-Generation```
 
 2. Install dependencies
    ```
    cd Object-Detection-Synthetic-Data-Generation
    pip3 install -r requirements.txt
    ```
 
 ## Creating input data
 
 Before you can start generating your synthetic data-set, you'll have to get some input images. You'll need to kinds of input images:
 - Foreground images
 - Background images
 
 For background images, you can take some pictures or download them from the internet (no additional processing needed). For the foreground images, you'll have to take pictures of the object and then remove the image's background.
 
 I recommend labeling a few images with LabelMe and then run my [create_input_images_from_labelme.py script](create_input_images_from_labelme.py) to extract the objects. For more information check out [create_input_images_with_labelme.md](doc/create_input_images_with_labelme.md).
 
 You can also make use of a tool look Photoshop or GIMP.
 - [Photoshop CC 2020: How To Remove a Background (Easiest Way)](https://www.youtube.com/watch?v=DWSa5SYzZu8)
 - [5 Ways To Remove A Background with GIMP](https://www.youtube.com/watch?v=lOzSiOIipSM)
 
 ## Generating images
 
 To generate images, run the [*data_generation.py*](data_generation.py) script.
 
```
usage: data_generation.py [-h] --input_dir INPUT_DIR --output_dir OUTPUT_DIR
                          [--augmentation_path AUGMENTATION_PATH]
                          --image_number IMAGE_NUMBER
                          [--max_objects_per_image MAX_OBJECTS_PER_IMAGE]
                          [--image_width IMAGE_WIDTH]
                          [--image_height IMAGE_HEIGHT]

Synthetic Image Generator

optional arguments:
  -h, --help            show this help message and exit
  --input_dir INPUT_DIR
                        Path to the input directory. It must contain a
                        backgrounds directory and a foregrounds directory
  --output_dir OUTPUT_DIR
                        The directory where images and label files will be
                        placed
  --augmentation_path AUGMENTATION_PATH
                        Path to albumentations augmentation pipeline file
  --image_number IMAGE_NUMBER
                        Number of images to create
  --max_objects_per_image MAX_OBJECTS_PER_IMAGE
                        Maximum number of objects per images
  --image_width IMAGE_WIDTH
                        Width of the output images
  --image_height IMAGE_HEIGHT
                        Height of the output images
```
 
 Example:
 `python data_generation.py --input_dir input/ --output_dir output/ --image_number 50`
 
 ## Converting output into other formats
 
 If the LabelMe format doesn't work for you, you can convert the data into another format.
 
 ### Convert to COCO
 
 You can convert the JSON files created by labelme to COCO using the [labelme2coco.py](https://github.com/Tony607/labelme2coco/blob/master/labelme2coco.py) file created by Github user [Tony607](https://github.com/Tony607).