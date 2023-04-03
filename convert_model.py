from exporters.coreml import export
from exporters.coreml.models import GPT2CoreMLConfig
from transformers import GPT2LMHeadModel, GPT2Tokenizer


"""
See:
https://github.com/huggingface/exporters
"""

model_ckpt = "gpt2"
base_model = GPT2LMHeadModel.from_pretrained(
    model_ckpt, torchscript=True
)
preprocessor = GPT2Tokenizer.from_pretrained(model_ckpt)

coreml_config = GPT2CoreMLConfig(
    base_model.config, 
    task="default",
)
mlmodel = export(
    preprocessor, base_model, coreml_config
)

mlmodel.save(f"models/{model_ckpt}.mlpackage")
