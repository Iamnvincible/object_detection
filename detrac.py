import os
import sys
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch
import collections
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


class Detrac(VisionDataset):
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

    def __init__(self, root, image_set='train', transform=None, target_transform=None, transforms=None, imgformat='png'):
        super(Detrac, self).__init__(
            root, transforms, transform, target_transform)
        valid_sets = ["train", "test"]
        base_dir = os.path.join(self.root, "")
        image_dir = os.path.join(base_dir, "JPEGImages")
        annotation_dir = os.path.join(base_dir, "Annotations")
        if not os.path.isdir(base_dir):
            raise RuntimeError('Dataset not found or corrupted.')
        splits_dir = os.path.join(base_dir, 'ImageSets')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n')+'.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x+"."+imgformat)
                       for x in file_names]
        self.annotations = [os.path.join(
            annotation_dir, x+".xml") for x in file_names]
        assert(len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())
        target = self.voc2coco(target)
        target['image_id'] = torch.tensor([index])
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text

        return voc_dict

    def voc2coco(self, voc_dict):
        coco_dict = {}
        labels = {'car': 1, 'van': 2, 'bus': 3, 'others': 4}
        boxes = []
        names = []
        if type(voc_dict['annotation']['object']) == list:
            objs = voc_dict['annotation']['object']
        else:
            objs = [voc_dict['annotation']['object']]

        for obj in objs:
            names.append(labels[obj['name']])
            x1 = int(obj['bndbox']['xmin'])
            y1 = int(obj['bndbox']['ymin'])
            x2 = int(obj['bndbox']['xmax'])
            y2 = int(obj['bndbox']['ymax'])
            boxes.append([x1, y1, x2, y2])
        coco_dict['boxes'] = torch.Tensor(boxes)
        coco_dict['labels'] = torch.LongTensor(names)
        tboxes = coco_dict['boxes']
        coco_dict['area'] = (
            tboxes[:, 3] - tboxes[:, 1]) * (tboxes[:, 2] - tboxes[:, 0])
        coco_dict['iscrowd'] = torch.zeros(
            (len(boxes,)), dtype=torch.int64)
        return coco_dict


if __name__ == "__main__":
    car = Detrac('D:\ISO\datasets\CARPK_devkit', 'train')
    print(len(car))
    print(car[0])
