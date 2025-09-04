# ---------------------------------------------------------------------------------------
# Qualcomm QAIC vLLM Example with BioMistral-7B
# ---------------------------------------------------------------------------------------
from vllm import LLM, SamplingParams
import random

# Sample prompts
prompts = [
    "What is the capital of France?",
    "Summarize the benefits of exercise.",
    "Explain quantum entanglement simply.",
] * 5  # repeat to simulate multiple batch requests

random.shuffle(prompts)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,   # greedy decoding
    max_tokens=128     # limit output length
)

# BioMistral-7B specific config
ctx_len = 4096   # BioMistral supports 4k context
seq_len = 1024   # capture sequence length (safe range)
decode_bsz = 4   # batch size for decode step, tune to your memory

# Create the LLM on QAIC
llm = LLM(
    model="BioMistral/BioMistral-7B",  # Hugging Face model ID
    device_group=[0],                  # first QAIC card
    max_num_seqs=decode_bsz,           # decode batch size
    max_model_len=ctx_len,             # maximum context length
    max_seq_len_to_capture=seq_len,    # sequence length captured per request
    quantization="mxfp6",              # preferred for QAIC
    kv_cache_dtype="mxint8",           # saves KV cache memory
    disable_log_stats=False,
    device="qaic",                     # force QAIC backend
    enable_prefix_caching=False,
    gpu_memory_utilization=1.0,
)

# Generate outputs
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    generated_tokens = output.outputs[0].token_ids
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print(f"Generated text: {generated_text}")
    print(f"Num generated tokens: {len(generated_tokens)}")
