import torch
import torchvision
from detrac import Detrac
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import os
import datetime
from engine import train_one_epoch, evaluate
import utils
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
# 1) 初始化
torch.distributed.init_process_group(backend="nccl")
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

root_dir = '/alihome/zrg/linjie/dataset'

carpklotdataset = Detrac(root_dir,
                         'train',
                         transform=transforms.Compose([transforms.ToTensor()]),
                         imgformat="jpg")
data_loader = DataLoader(carpklotdataset,
                         batch_size=8,
                         shuffle=True,
                         num_workers=2,
                         collate_fn=utils.collate_fn)
rand_loader = DataLoader(dataset=carpklotdataset,
                         batch_size=8,
                         sampler=DistributedSampler(carpklotdataset))
carpklotdataset_test = Detrac(root_dir,
                              'test',
                              transform=transforms.Compose(
                                  [transforms.ToTensor()]),
                              imgformat="jpg")
data_loader_test = DataLoader(carpklotdataset_test,
                              batch_size=1,
                              shuffle=False,
                              collate_fn=utils.collate_fn)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=5,
                                                             pretrained=False)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[8, 11],
                                                    gamma=0.1)
is_resume = True
resume_path = None
if is_resume and resume_path:
    print('Resume training')
    checkpoint = torch.load(resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
model.to(device)
print("Let's use", torch.cuda.device_count(), "GPUs!")
model = torch.nn.parallel.DistributedDataParallel(model,
                                                  device_ids=[local_rank],
                                                  output_device=local_rank)
# model = model.cuda()
# model = nn.DataParallel(model)  # multi-GPU
# device = torch.device("cuda:0")
# model.to(device)
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
