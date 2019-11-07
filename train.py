import torch
import torchvision
from carpk import Carpk
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import os
import datetime
from engine import train_one_epoch, evaluate
import utils

root_dir = '/alihome/zrg/linjie/dataset'

carpklotdataset = Carpk(root_dir,
                        'train',
                        transform=transforms.ToTensor(),
                        imgformat="jpg")
data_loader = DataLoader(carpklotdataset,
                         batch_size=2,
                         shuffle=True,
                         num_workers=4,
                         collate_fn=utils.collate_fn)
carpklotdataset_test = Carpk(root_dir,
                             'test',
                             transform=transforms.ToTensor(),
                             imgformat="jpg")
data_loader_test = DataLoader(carpklotdataset_test,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=utils.collate_fn)

is_resume = True
resume_path = None
if is_resume and resume_path:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        num_classes=2, pretrained=False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.02,
                                momentum=0.9,
                                weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[8, 11],
                                                        gamma=0.1)
else:
    print('Resume training')
    checkpoint = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
device = torch.device("cuda")
model.to(device)
# Training
print('Start training')
start_time = time.time()
for epoch in range(10):
    train_one_epoch(model, optimizer, data_loader, device, epoch, 50)
    lr_scheduler.step()
    if True:
        utils.save_on_master(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join("./", 'model_{}.pth'.format(epoch)))

    # evaluate after every epoch
    evaluate(model, data_loader_test, device=device)

total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print('Training time {}'.format(total_time_str))
