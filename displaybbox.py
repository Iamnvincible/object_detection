import torch
import torchvision
from PIL import ImageDraw
from carpk import Carpk


def getbbox(objects):
    bbox = list()
    islist = type(objects) == list
    if islist:
        for obj in objects:
            bndbox = obj['bndbox']
            xmin = int(bndbox['xmin'])
            ymin = int(bndbox['ymin'])
            xmax = int(bndbox['xmax'])
            ymax = int(bndbox['ymax'])
            box = [xmin, ymin, xmax, ymax]
            bbox.append(box)
    else:
        bndbox = objects['bndbox']
        xmin = int(bndbox['xmin'])
        ymin = int(bndbox['ymin'])
        xmax = int(bndbox['xmax'])
        ymax = int(bndbox['ymax'])
        box = [xmin, ymin, xmax, ymax]
        bbox.append(box)
    return torch.tensor(bbox)


def drawbbox(img, bboxes):
    if not img or len(bboxes) == 0:
        return
    draw = ImageDraw.Draw(img)
    for item in bboxes:
        draw.rectangle(item.tolist(), outline='red', width=3)
    # display(img) #work for IPython
    img.show()


def showgt(imgtuple):
    img = imgtuple[0]
    bboxes = getbbox(imgtuple[1]['annotation']['object'])
    drawbbox(img, bboxes)


if __name__ == "__main__":
    choose = False
    if choose:
        root = "D:\dataset"
        voc = torchvision.datasets.VOCDetection(root,
                                                year='2007',
                                                image_set='train',
                                                download=False,
                                                transform=None,
                                                target_transform=None,
                                                transforms=None)
        showgt(voc[118])
    else:
        car = Carpk('D:\ISO\datasets\CARPK_devkit', 'train')
        showgt(car[1])
