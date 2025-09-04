Install from source (bare-metal or your own container) 

1) System prep (Ubuntu host with AI100s)
   
# Make sure Platform SDK is installed (NOT App SDK).
# After installation, you should have /opt/qti-aic and /dev/qaic devices.

# Give your user device access
sudo usermod -aG qaic $USER && newgrp qaic

# Create a clean Python 3.10 venv
python3.10 -m venv qaic-vllm-venv
source qaic-vllm-venv/bin/activate

# Basic pip hygiene
pip install -U pip wheel setuptools
2) Install Efficient-Transformers (required by QAIC vLLM flow)
Qualcomm’s docs call this out explicitly. quic.github.io

pip install "git+https://github.com/quic/efficient-transformers@release/v1.20.0"

3) Get vLLM and apply QAIC patch from the SDK
# Clone vLLM (use the version Qualcomm notes)
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.8.5

# Apply Qualcomm’s QAIC backend patch shipped with the Platform SDK
git apply /opt/qti-aic/integrations/vllm/qaic_vllm.patch

# Tell vLLM we’re building for QAIC
export VLLM_TARGET_DEVICE=qaic

# Install vLLM editable
pip install -e .

(Those exact steps are mirrored in the guide’s “Installing from source” section.) quic.github.io
4) Quick example to verify QAIC runtime
python /vllm/examples/offline_inference/qaic.py


You should see a short generation and device timings, confirming the QAIC path is active. quic.github.io
5) Serve BioMistral-7B
Same as in Option A, just run the API server from your venv:
export HF_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # if needed
export VLLM_TARGET_DEVICE=qaic

python -m vllm.entrypoints.openai.api_server \
  --model BioMistral/BioMistral-7B \
  --dtype half \
  --max-model-len 4096 \
  --enforce-eager

Query it:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BioMistral/BioMistral-7B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 32
  }'

Tips & gotchas
    • You no longer have to hand-feed KV buffers or io-order files. vLLM+QAIC abstracts the runner details. (Those earlier “File size greater than buffer size” errors disappear with this stack.)
    • Device selection:
        ◦ QAIC_VISIBLE_DEVICES="0,1" (env var) limits which AI100s the QAIC backend sees.
        ◦ Make sure /dev/qaic* devices are present in the container (--device /dev/qaic) and your user is in the qaic group.
    • First-run compile time: the very first load of a new model may spend time exporting/compiling. Subsequent runs reuse artifacts.
    • Prefix caching: Supported with QAIC in vLLM, but with limitations (no paged-attention sharing, etc.). See the doc’s section and example:
