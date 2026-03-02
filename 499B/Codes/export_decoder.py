import torch
from transformers import OPTForCausalLM
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

print("Loading OPT-1.3b...")
model = OPTForCausalLM.from_pretrained('facebook/opt-1.3b', torch_dtype=torch.float32)
model.eval()
print("Model loaded!")

os.makedirs('onnx_models', exist_ok=True)
dummy_ids = torch.zeros(1, 32, dtype=torch.long)

print("Exporting to ONNX... (10-15 min)")
torch.onnx.export(
    model, (dummy_ids,),
    'onnx_models/opt_decoder.onnx',
    input_names=['input_ids'],
    output_names=['logits'],
    dynamic_axes={'input_ids': {0:'batch',1:'seq'}, 'logits': {0:'batch',1:'seq'}},
    opset_version=14
)
print("Export done!")

print("Quantizing to INT8...")
quantize_dynamic('onnx_models/opt_decoder.onnx', 'onnx_models/opt_decoder_int8.onnx', weight_type=QuantType.QInt8)
size = os.path.getsize('onnx_models/opt_decoder_int8.onnx')/1024/1024
print(f"INT8 size: {size:.1f} MB")
print("All done!")
