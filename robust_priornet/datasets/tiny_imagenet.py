import io
import os
import tarfile

import torchvision.datasets as datasets
from PIL import Image


class TinyImageNet(datasets.VisionDataset):
    """
    Only loads the val dataset from the downloaded tar file.
    """
    def __init__(self, root, transform=None, target_transform=None, split='test', **kwargs):
        super(TinyImageNet, self).__init__(root)
        assert split in ['train', 'test']
        self.transform = transform
        self.target_transform = target_transform
        self.tf = tarfile.open(os.path.join(root, 'tiny-imagenet.tar.gz'))
        self.classes = self._get_classnames_idx_dict(root)
        if split == 'test':
            self.valset_image_dir = os.path.join(root, 'tiny-imagenet-200', 'val', 'images')
            self.val_info = self.tf.extractfile(os.path.join(root,
                                                             'tiny-imagenet-200',
                                                             'val',
                                                             'val_annotations.txt')).readlines()
            images_labels = {}
            for i in range(len(self.val_info)):
                line = self.val_info[i].decode("utf-8").split()
                images_labels[line[0]] = line[1]
            self.images = self._get_image_path_classidx_tuples(root,
                                                               os.path.join('tiny-imagenet-200',
                                                                            'val',
                                                                            'images'),
                                                               images_labels)
        elif split == 'train':
            raise NotImplementedError

    def _get_classnames_idx_dict(self, root):
        class_list = self.tf.extractfile(os.path.join(root,
                                                      'tiny-imagenet-200',
                                                      'wnids.txt')).readlines()
        class_list = [class_list[i].decode("utf-8").split()[0] for i in range(len(class_list))]
        class_list.sort()
        class_to_idx = {class_list[i]: i for i in range(len(class_list))}
        return class_to_idx

    def _get_image_path_classidx_tuples(self, root, base_path, image_label_dict):
        images = []
        for _, img_name in enumerate(image_label_dict):
            path = os.path.join(root, base_path, img_name)
            classidx = self.classes[image_label_dict[img_name]]
            img = Image.open(io.BytesIO(self.tf.extractfile(path).read()), mode='r')
            images.append((img, classidx))
        return images

    def __getitem__(self, index):
        img, target = self.images[index]
        #try:
        # img = Image.open(io.BytesIO(self.tf.extractfile(img_path).read()), mode='r')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        #except Exception:
        #    print(f"Failed at index: {index}, img_path: {img_path}")

        if index == (self.__len__() - 1):  # close tarfile opened in __init__
            self.tf.close()
        return img, target

    def __len__(self):
        return len(self.images)
