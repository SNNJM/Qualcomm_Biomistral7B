# QuickStart Note: this version uses python 3.10

Installation 
1. Create venv 
2. Install platform: cd platform/deb for debian ./install.sh
3. sudo ./uninstall if necessary & sync

How to run: 
1. python3 converter.py , this will get Biomistral7B files and convert to onnx
2. python3 patch_dimname.py, this to patch (fix error in compiling with qaic)
3. Compile with Qualcomm: /opt/qti-aic/exec/qaic-exec -m=/home/ai/Desktop/Qualcomm/models/Biomistral7B/biomistral-onnx2/model.onnx -aic-hw -aic-hw-version=2.0 -convert-to-fp16 -aic-num-cores=16 -mos=1 -ols=16 -onnx-define-symbol=batch_size,1 -onnx-define-symbol=past_sequence_length,1 -onnx-define-symbol=sequence_length,1 -onnx-define-symbol='past_sequence_length + sequence_length',2 -onnx-define-symbol='past_sequence_length + 1',2 -aic-binary-dir=/home/ai/Desktop/Qualcomm/models/Qbiomistral -compile-only
  Output: Reading ONNX Model from /home/ai/Desktop/Qualcomm/models/Biomistral7B/biomistral-onnx2/model.onnx
  Compile started ............... 
  Compiling model with FP16 precision.
  Generated binary is present at /home/ai/Desktop/Qualcomm/models/Qbiomistral
4. python3 dummy_input.py , create dummy input data
5. mkdir -p out
/opt/qti-aic/exec/qaic-runner \
  -t /home/ai/Desktop/Qualcomm/models/Qbiomistral \
  --bound-random \
  --num-iter 1 \
  --write-output-dir ./out \
  --write-output-num-samples 1

   output: Using bounded random inputs
  Writing file:./out/logits-activation-0-inf-0.bin
  Writing file:./out/present.0.key-activation-0-inf-0.bin
  Writing file:./out/present.0.value-activation-0-inf-0.bin
  Writing file:./out/present.1.key-activation-0-inf-0.bin
  Writing file:./out/present.1.value-activation-0-inf-0.bin
  Writing file:./out/present.2.key-activation-0-inf-0.bin
  Writing file:./out/present.2.value-activation-0-inf-0.bin
  Writing file:./out/present.3.key-activation-0-inf-0.bin
  Writing file:./out/present.3.value-activation-0-inf-0.bin
  Writing file:./out/present.4.key-activation-0-inf-0.bin
  Writing file:./out/present.4.value-activation-0-inf-0.bin
  Writing file:./out/present.5.key-activation-0-inf-0.bin
  Writing file:./out/present.5.value-activation-0-inf-0.bin
  Writing file:./out/present.6.key-activation-0-inf-0.bin
  Writing file:./out/present.6.value-activation-0-inf-0.bin
  Writing file:./out/present.7.key-activation-0-inf-0.bin
  Writing file:./out/present.7.value-activation-0-inf-0.bin
  Writing file:./out/present.8.key-activation-0-inf-0.bin
  Writing file:./out/present.8.value-activation-0-inf-0.bin
  Writing file:./out/present.9.key-activation-0-inf-0.bin
  Writing file:./out/present.9.value-activation-0-inf-0.bin
  Writing file:./out/present.10.key-activation-0-inf-0.bin
  Writing file:./out/present.10.value-activation-0-inf-0.bin
  Writing file:./out/present.11.key-activation-0-inf-0.bin
  Writing file:./out/present.11.value-activation-0-inf-0.bin
  Writing file:./out/present.12.key-activation-0-inf-0.bin
  Writing file:./out/present.12.value-activation-0-inf-0.bin
  Writing file:./out/present.13.key-activation-0-inf-0.bin
  Writing file:./out/present.13.value-activation-0-inf-0.bin
  Writing file:./out/present.14.key-activation-0-inf-0.bin
  Writing file:./out/present.14.value-activation-0-inf-0.bin
  Writing file:./out/present.15.key-activation-0-inf-0.bin
  Writing file:./out/present.15.value-activation-0-inf-0.bin
  Writing file:./out/present.16.key-activation-0-inf-0.bin
  Writing file:./out/present.16.value-activation-0-inf-0.bin
  Writing file:./out/present.17.key-activation-0-inf-0.bin
  Writing file:./out/present.17.value-activation-0-inf-0.bin
  Writing file:./out/present.18.key-activation-0-inf-0.bin
  Writing file:./out/present.18.value-activation-0-inf-0.bin
  Writing file:./out/present.19.key-activation-0-inf-0.bin
  Writing file:./out/present.19.value-activation-0-inf-0.bin
  Writing file:./out/present.20.key-activation-0-inf-0.bin
  Writing file:./out/present.20.value-activation-0-inf-0.bin
  Writing file:./out/present.21.key-activation-0-inf-0.bin
  Writing file:./out/present.21.value-activation-0-inf-0.bin
  Writing file:./out/present.22.key-activation-0-inf-0.bin
  Writing file:./out/present.22.value-activation-0-inf-0.bin
  Writing file:./out/present.23.key-activation-0-inf-0.bin
  Writing file:./out/present.23.value-activation-0-inf-0.bin
  Writing file:./out/present.24.key-activation-0-inf-0.bin
  Writing file:./out/present.24.value-activation-0-inf-0.bin
  Writing file:./out/present.25.key-activation-0-inf-0.bin
  Writing file:./out/present.25.value-activation-0-inf-0.bin
  Writing file:./out/present.26.key-activation-0-inf-0.bin
  Writing file:./out/present.26.value-activation-0-inf-0.bin
  Writing file:./out/present.27.key-activation-0-inf-0.bin
  Writing file:./out/present.27.value-activation-0-inf-0.bin
  Writing file:./out/present.28.key-activation-0-inf-0.bin
  Writing file:./out/present.28.value-activation-0-inf-0.bin
  Writing file:./out/present.29.key-activation-0-inf-0.bin
  Writing file:./out/present.29.value-activation-0-inf-0.bin
  Writing file:./out/present.30.key-activation-0-inf-0.bin
  Writing file:./out/present.30.value-activation-0-inf-0.bin
  Writing file:./out/present.31.key-activation-0-inf-0.bin
  Writing file:./out/present.31.value-activation-0-inf-0.bin
   ---- Stats ----
  InferenceCnt 1 TotalDuration 128510us BatchSize 1 Inf/Sec 7.781


 


