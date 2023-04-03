import sys

import coremltools as ct
from loguru import logger
from transformers import AutoTokenizer, GPT2TokenizerFast, GPT2LMHeadModel

from .utils import timer


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])

# prevent "Disabling parallelism to avoid deadlocks" warning from huggingface/tokenizers
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


MODEL_NAME = "gpt2"


def load_coreml(model_path):
    """
    Expects an .mlmodel file, manually downloaded.
    """
    logger.debug("Loading CoreML model...")
    with timer() as timing:
        model = ct.models.MLModel(model_path)
    logger.debug(f"Loaded CoreML model in {timing.execution_time_ns / 1e6:.2f}ms")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return model, tokenizer


def load_pytorch(model_name: str = "gpt2"):
    logger.debug(f"Loading PyTorch model '{model_name}'...")
    with timer() as timing:
        model = GPT2LMHeadModel.from_pretrained(model_name)
    logger.debug(f"Loaded PyTorch model in {timing.execution_time_ns / 1e6:.2f}ms")

    logger.debug("Loading tokenizer...")
    with timer() as timing:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    logger.debug(f"Loaded tokenizer in {timing.execution_time_ns / 1e6:.2f}ms")

    return model, tokenizer
