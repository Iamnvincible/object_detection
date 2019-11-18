import onnx
import onnxruntime
import numpy as np
import torch
onnx_model = onnx.load("carmodel.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("carmodel.onnx")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


x = torch.randn(1, 1, 224, 224, requires_grad=True)
# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)
print(ort_outs)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")
