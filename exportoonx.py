import torch
import torchvision
checkpoint = torch.load(
    r"D:\dataset\UA-DETRAC\model_9.pth", map_location='cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    num_classes=5, pretrained=False)
model.load_state_dict(checkpoint['model'])
x = torch.rand(2, 3, 720, 1280)
torch.onnx.export(model, args=x, f="carmodel.onnx", verbose=True)
