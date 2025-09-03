# convert_biomistral_ort_v2.py  (Optimum 1.27-friendly)
import argparse, onnx
from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--model_id", default="BioMistral/BioMistral-7B")
p.add_argument("--outdir",   default="biomistral-onnx")
p.add_argument("--trust_remote_code", action="store_true")
args = p.parse_args()

outdir = Path(args.outdir).resolve()
outdir.mkdir(parents=True, exist_ok=True)

# Export HF → ONNX (no 'opset' here on 1.27)
model = ORTModelForCausalLM.from_pretrained(
    args.model_id,
    export=True,
    trust_remote_code=args.trust_remote_code,
    provider="CPUExecutionProvider",
)
model.save_pretrained(outdir)

tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
tok.save_pretrained(outdir)

def show_io(path: Path):
    if not path.exists():
        print("Missing:", path); return
    print(f"\n== {path.name} ==")
    m = onnx.load(str(path))
    def dims(vt): return [d.dim_param or d.dim_value for d in vt.type.tensor_type.shape.dim]
    print("Inputs:");  [print(" ", i.name, dims(i)) for i in m.graph.input]
    print("Outputs:"); [print(" ", o.name, dims(o)) for o in m.graph.output]

prefill = outdir / "decoder_model.onnx"
decode  = outdir / "decoder_with_past_model.onnx"
show_io(prefill)
show_io(decode)

print("\nWrote to", outdir)
for f in sorted(outdir.iterdir()):
    print(" ", f.name)

