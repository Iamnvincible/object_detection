from torchvision.datasets.vision import VisionDataset
from detrac import Detrac
from carpk import Carpk


class DatasetUnion(VisionDataset):
    def __init__(self,
                 roots,
                 setformations,
                 imgformats,
                 image_set='train',
                 transforms=None,
                 transform=None,
                 target_transform=None):
        super().__init__(roots[0],
                         transforms=transforms,
                         transform=transform,
                         target_transform=target_transform)
        self.images = []
        self.annotations = []
        datasets = ['carpk', 'pucpr', 'detrac']
        image_sets = ['train', 'val', 'test']
        for index, imgformation in enumerate(setformations):
            if imgformation in datasets:
                if imgformation == datasets[0]:
                    self.carpk = Carpk(roots[index], image_set, transform,
                                       target_transform, transforms,
                                       imgformats[index])
                if imgformation == datasets[1]:
                    self.pucpr = Carpk(roots[index], image_set, transform,
                                       target_transform, transforms,
                                       imgformats[index])
                if imgformation == datasets[2]:
                    self.detrac = Detrac(roots[index], image_set, transform,
                                         target_transform, transforms,
                                         imgformats[index])
        if self.carpk:
            self.images.append(self.carpk.images)
        if self.pucpr:
            self.images.append(self.pucpr.images)
        if self.detrac:
            self.images.append(self.detrac.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.carpk and index < len(self.carpk):
            return self.carpk[index]
        return self.detrac[index - len(self.carpk)]
