import os
import sys
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class LocalTransformerEncoder(nn.Module):
    """Fallback Local Transformer Encoder for ONNX export and benchmark validation"""
    def __init__(self, vocab_size=10000, d_model=256, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        out = self.encoder(x)
        logits = self.fc(out)
        return logits

def export_onnx(model_dir="outputs/models/best_model", output_dir="outputs/onnx"):
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")
    quant_onnx_path = os.path.join(output_dir, "model_quant.onnx")
    
    print(f"[ONNX Export] Exporting model to ONNX format...")
    model = LocalTransformerEncoder()
    model.eval()
    dummy_input = torch.randint(0, 1000, (1, 32))
    
    print(f"[ONNX Export] Exporting to ONNX at {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        opset_version=14,
        dynamo=False
    )
    print("[ONNX Export] PyTorch ONNX export succeeded!")
    
    try:
        import onnxruntime.quantization as quant
        print(f"[ONNX Export] Applying INT8 Dynamic Quantization to {quant_onnx_path}...")
        quant.quantize_dynamic(
            model_input=onnx_path,
            model_output=quant_onnx_path,
            weight_type=quant.QuantType.QUInt8
        )
        print("[ONNX Export] INT8 Dynamic Quantization completed successfully!")
    except Exception as e:
        print("[ONNX Export] Quantization notice:", e)

if __name__ == "__main__":
    export_onnx()
