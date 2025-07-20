import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.exporters.openvino import export_models
from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
from openvino import save_model
from openvino_tokenizers import convert_tokenizer
from huggingface_hub import login


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
INT4_DIR = "models/llama3_int4"
HF_TOKEN = os.environ.get('HF_TOKEN','')


# output directories
INT4_DIR = Path("models/llama3_int4")

def quantize_fp16_to_int4(int4_dir: Path):
    '''
    Quantizes FP16 Llama-3.1-8B-Instruct model using 4-bit weights-only quantization.
    For reference see below: 
    https://huggingface.co/docs/optimum/main/intel/openvino/optimization#4-bit
    https://docs.openvino.ai/2025/openvino-workflow-generative/ov-tokenizers.html
    '''
    # Configure 4-bit weight-only compression
    q_config = OVWeightQuantizationConfig(bits=4)
    # Load and compress model
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
    model = OVModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,
        quantization_config=q_config
    )
    
    # Save compressed model
    model.save_pretrained(str(int4_dir))
    save_model(ov_tokenizer, f"{str(int4_dir)}/openvino_tokenizer.xml")
    save_model(ov_detokenizer, f"{str(int4_dir)}/openvino_detokenizer.xml")
    tokenizer.save_pretrained(str(int4_dir))
    print(f"Saved INT4-weight compressed model to: {int4_dir}")


if __name__ == "__main__":
    os.makedirs(INT4_DIR, exist_ok=True)
    login(token=HF_TOKEN)
    quantize_fp16_to_int4(INT4_DIR)
