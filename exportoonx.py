import torch
import torchvision
model_path = r'd:\dataset\UA-DETRAC\model_9.pth'
checkpoint = torch.load(
    model_path, map_location='cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    num_classes=5, pretrained=False)
model.load_state_dict(checkpoint['model'])
model.eval()
x = torch.rand(1, 3, 720, 1280)
torch.onnx.export(model, args=x, f="carmodel2.onnx",
                  do_constant_folding=True, verbose=True, opset_version=11)
