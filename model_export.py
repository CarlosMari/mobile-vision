from mobile_cv.model_zoo.models.fbnet_v2 import fbnet
from mobile_cv.model_zoo.models.preprocess import get_preprocess
import functools
import torch
import requests
from io import BytesIO
from PIL import Image

from torch.onnx import dynamo_export


torch.load = functools.partial(torch.load, weights_only=False)
torch.load = torch.__dict__["load"]

model_name = "fbnet_a"
model = fbnet(model_name, pretrained=True)
model.eval()


# 2. dummy input (batch=1, 3x224x224)
dummy_input = torch.randn(1, 3, 224, 224, device="cpu")

# 3. export
'''torch.onnx.export(
    model, 
    dummy_input, 
    "fbnet_a.onnx",
    export_params=True,
    opset_version=13,    # Use >=11
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    #dynamo=True,
)'''
#exported_program = dynamo_export(model, dummy_input)
#exported_program.save("fbnet_a.onnx")


#torch.onnx.export(exported_program, dummy_input, "fbnet_a.onnx")
# OR go back to regular export:


# Try with verbose mode to see what's happening
torch.onnx.export(
    model, 
    dummy_input, 
    "fbnet_a.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    verbose=True,  # Add this to see progress
)