import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def persist_images(images, mean, std, n_channels, image_dir):
    assert type(images) == np.ndarray, ("Unexpected input type. Cannot be persisted.")
    # Images to be in [-1, 1] interval, so rescale them back to [0, 255].
    mean = np.array(mean).reshape((3, 1, 1))
    std = np.array(std).reshape((3, 1, 1))
    images = np.asarray((images*std + mean)*255.0, dtype=np.uint8)

    for i, image in enumerate(images):
        if n_channels == 1:
            # images were added new channels (3) to go through VGG, so remove unnecessary channels
            Image.fromarray(image[0, :, :]).save(os.path.join(image_dir, f"{i}.png"))
        else:
            Image.fromarray(image).save(os.path.join(image_dir, f"{i}.png"))

def persist_image_dataset(dataset, mean, std, n_channels, image_dir):
    assert isinstance(dataset, Dataset), ("Dataset is not of right type. Cannot be persisted.")
    if len(dataset) > 0:
        images = []
        testloader = DataLoader(dataset, batch_size=128,
                                shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for _, data in enumerate(testloader, 0):
                image, _ = data
                images.append(image)

        persist_images(torch.cat(images, dim=0).numpy(), mean, std, n_channels, image_dir)
