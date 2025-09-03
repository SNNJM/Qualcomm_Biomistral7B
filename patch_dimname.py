import onnx, sys
model_path = sys.argv[1]
old = "past_sequence_length + sequence_length"
new = "total_sequence_length"

m = onnx.load(model_path)
changed = 0

def patch_value_info(vi):
    global changed
    if vi.type.HasField("tensor_type"):
        for d in vi.type.tensor_type.shape.dim:
            if d.dim_param == old:
                d.dim_param = new
                changed += 1

for vi in list(m.graph.input) + list(m.graph.output) + list(m.graph.value_info):
    patch_value_info(vi)

for init in m.graph.initializer:
    # initializers have concrete sizes; usually not symbolic
    pass

onnx.save(m, model_path)
print(f"Patched {changed} dimension(s). Saved back to {model_path}.")
