import onnx
import onnxruntime
import numpy as np
import torch
import torchvision
from detrac import Detrac
import dataset.transforms as T

root = r"D:\dataset\UA-DETRAC\Detrac_dataset"
transforms = []
transforms.append(T.ToTensor())
transformscompose = T.Compose(transforms)
detrac = Detrac(root, imgformat='jpg', transforms=transformscompose)
img = [detrac[0][0]]

onnx_model = onnx.load("carmodel2.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("carmodel2.onnx")

checkpoint = torch.load(
    r"D:\dataset\UA-DETRAC\model_9.pth", map_location='cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    num_classes=5, pretrained=False)
model.load_state_dict(checkpoint['model'])
model.eval()
torch_out = model(img)
print(torch_out)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


x = detrac[0][0].view(1, 3, 720, 1280).numpy()

# print(type(x))
# x = torch.randn(1, 3, 960, 540, requires_grad=True)
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)
# np.testing.assert_allclose(
#     to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
#print("Exported model has been tested with ONNXRuntime, and the result looks good!")
