import os
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch


class Carpk(VisionDataset):
    """
        Args:
            root (string): Root directory of the VOC Dataset.
            image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``

            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, required): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
    """

    def __init__(self, root, image_set='train', transform=None, target_transform=None, transforms=None):
        super(Carpk, self).__init__(
            root, transforms, transform, target_transform)
        valid_sets = ["train", "test"]
        base_dir = os.path.join(self.root, "data")
        image_dir = os.path.join(base_dir, "Images")
        annotation_dir = os.path.join(base_dir, "Annotations")
        if not os.path.isdir(base_dir):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(base_dir, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n')+'.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x+".png") for x in file_names]
        self.annotations = [os.path.join(
            annotation_dir, x+".txt") for x in file_names]
        assert(len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_carpk_txt(self.annotations[index])
        #target['annotation']['filename'] = self.images[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target['annotation']

    def __len__(self):
        return len(self.images)

    def parse_carpk_txt(self, note):
        car_dict = {}
        car_dict['annotation'] = {}
        objects = []
        with open(note, 'r') as f:
            targets = [x.strip() for x in f.readlines()]
        for item in targets:
            ordinates = item.split(' ')
            vocformat = False
            if vocformat:
                obj_dict = {}
                obj_dict['bndbox'] = {}
                obj_dict['bndbox']['xmin'] = ordinates[0]
                obj_dict['bndbox']['ymin'] = ordinates[1]
                obj_dict['bndbox']['xmax'] = ordinates[2]
                obj_dict['bndbox']['ymax'] = ordinates[3]
                obj_dict['name'] = ordinates[4]
                objects.append(obj_dict)
            else:
                objects.append([int(ordinates[0]), int(ordinates[1]),
                                int(ordinates[2]), int(ordinates[3])])

        car_dict['annotation']['boxes'] = torch.Tensor(objects)
        car_dict['annotation']['labels'] = torch.ones(
            (len(objects,)), dtype=torch.int64)
        return car_dict


if __name__ == "__main__":
    car = Carpk('D:\ISO\datasets\CARPK_devkit', 'train')
    print(len(car))
    print(car[0])
