"""
Contains builder class for composing many image transformations available in torchvision.
"""
import torchvision.transforms as transforms
from PIL import Image


class TransformsBuilder:
    """
    Builder class for adding any number of transforms, finally composes them together
    using torchvision.transforms.Compose().
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._transforms_list = []

    def get_transforms(self):
        return transforms.Compose(self._transforms_list)

    def add_to_tensor(self):
        self._transforms_list.append(transforms.ToTensor())

    def add_normalize(self, mean, std):
        self._transforms_list.append(transforms.Normalize(mean, std))

    def add_resize(self, target_img_size):
        self._transforms_list.append(transforms.Resize(target_img_size, Image.BICUBIC))

    def add_center_crop(self, target_img_size):
        self._transforms_list.append(transforms.CenterCrop(target_img_size))

    def add_random_crop(self, target_img_size):
        self._transforms_list.append(transforms.RandomCrop(target_img_size))

    def add_rgb_channels(self, num_channels):
        if num_channels < 3:
            self._transforms_list.append(transforms.Grayscale(num_output_channels=3))

    def add_padding(self, pad_size):
        self._transforms_list.append(transforms.Pad(pad_size, padding_mode='reflect'))

    def add_random_flipping(self):
        self._transforms_list.append(transforms.RandomHorizontalFlip())

    def add_rotation(self, rotation_angle):
        self._transforms_list.append(transforms.RandomRotation(degrees=rotation_angle,
                                                               resample=Image.BICUBIC))
