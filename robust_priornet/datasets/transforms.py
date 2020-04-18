import torchvision.transforms as transforms
from PIL import Image

class TransformsBuilder:
    """
    Builder class for adding any number of transforms, finally composes them together using torchvision.transforms.Compose().
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._transforms_list = []

    def get_transforms(self):
        return transforms.Compose(self._transforms_list)
    
    def add_to_tensor(self):
        self._transforms_list.append(transforms.ToTensor())
    
    def add_Normalize(self, mean, std):
        self._transforms_list.append(transforms.Normalize(mean, std))

    def add_resize(self, target_img_size):
        self._transforms_list.append(transforms.Resize(target_img_size, Image.BICUBIC))
        self._transforms_list.append(transforms.CenterCrop(target_img_size))