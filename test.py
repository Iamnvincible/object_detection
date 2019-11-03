# from detrac import Detrac
# import torch
# import torchvision
# from PIL import ImageDraw
# dataset = Detrac(r'D:\dataset\UA-DETRAC\Detrac_dataset',
#                  'train', imgformat='jpg')
# for index in range(10000):
#     dataset[index]
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# 输入维度
input_size = 5
# 输出维度
output_size = 2
# 单批大小
batch_size = 80
# 数据总数
data_size = 160
# 设备指向
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RandomDataset(Dataset):
    # 数据集，生成input_size*data_size维向量

    def __init__(self, length, size):
        self.len = length
        self.data = torch.randn(length, size)  # datasize*input_size

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(data_size, input_size),
                         batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("In Model: input size", input.size(),
              "output size", output.size())

        return output


# 初始化模型
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # 将模型分布到所有GPU上
    model = nn.DataParallel(model)
#模型必须在gpu0上
model.to(device)
# 训练
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
