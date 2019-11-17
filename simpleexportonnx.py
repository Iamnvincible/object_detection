import torch
import torchvision
device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=False).to(device)
model.eval()

image = torch.randn(3, 800, 1333, device=device)
images = [image]

torch.onnx.export(model, (images, ), 'faster_rcnn.onnx',
                  do_constant_folding=True, opset_version=11)
