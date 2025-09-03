import numpy as np

inputs = {
  "input_ids": np.array([[42]], dtype=np.int64),
  "attention_mask": np.ones((1,2), dtype=np.int64),
  "position_ids": np.array([[0]], dtype=np.int64),
}

# 32 layers of cache, each [1, 8, 1, 128]
for l in range(32):
    inputs[f"past_key_values.{l}.key"] = np.zeros((1,8,1,128), dtype=np.float16)
    inputs[f"past_key_values.{l}.value"] = np.zeros((1,8,1,128), dtype=np.float16)

np.savez("biomistral_inputs.npz", **inputs)
