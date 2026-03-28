import sys
sys.path.insert(0, 'C:\\pylib') 
import onnx
from onnx import helper, TensorProto
import os

def convert_model_to_fp16(input_path, output_path):
    model = onnx.load(input_path)
    
    # 1. Convert all constant weights (Initializers) to float16
    for i in range(len(model.graph.initializer)):
        tensor = model.graph.initializer[i]
        if tensor.data_type == TensorProto.FLOAT:
            from onnx import numpy_helper
            arr = numpy_helper.to_array(tensor).astype('float16')
            new_tensor = numpy_helper.from_array(arr, tensor.name)
            model.graph.initializer[i].CopyFrom(new_tensor)

    # 2. Update Input and Output types to float16
    for input_node in model.graph.input:
        if input_node.type.tensor_type.elem_type == TensorProto.FLOAT:
            input_node.type.tensor_type.elem_type = TensorProto.FLOAT16
            
    for output_node in model.graph.output:
        if output_node.type.tensor_type.elem_type == TensorProto.FLOAT:
            output_node.type.tensor_type.elem_type = TensorProto.FLOAT16

    # 3. Update all internal node types (Value Info)
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT16

    onnx.save(model, output_path)
    print(f"FP16 Success! Size: {os.path.getsize(output_path)/(1024*1024):.2f} MB")

convert_model_to_fp16("mobilenet_v2.onnx", "mobilenet_v2_fp16.onnx")