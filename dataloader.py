
from carpk import Carpk
from torch.utils.data import DataLoader
from torchvision import transforms
carpklotdataset = Carpk('D:\ISO\datasets\CARPK_devkit', 'train', transform=transforms.Compose(
    [transforms.Resize(256), transforms.ToTensor()]))
print(len(carpklotdataset))
