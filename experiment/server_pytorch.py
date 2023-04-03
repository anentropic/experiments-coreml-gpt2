import argparse
import multiprocessing as mp
import sys
from dataclasses import dataclass
from multiprocessing.connection import Connection

import torch
from loguru import logger
from termcolor import colored, cprint

from .loader import load_pytorch
from .utils import timer


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])


@dataclass
class CompletionConfig:
    model: str = "gpt2"
    temperature: float = 0.85
    top_k: int = 50
    top_p: float = 0.99
    max_length: int = 140
    repetition_penalty: float = 1.0


def child_process(
    conn: Connection,
    config: CompletionConfig,
):
    try:
        model, tokenizer = load_pytorch(config.model)
    except:
        import traceback
        traceback.print_exc()
        conn.send(False)
        conn.close()
        return
    else:
        # Signal to the parent process that we are ready to receive inputs
        conn.send(True)

    while True:
        input_val = conn.recv()

        # If the input is None, exit the loop and terminate the child process
        if input_val is None:
            break

        if isinstance(input_val, CompletionConfig):
            config = input_val
            continue

        logger.debug("Tokenizing input...")
        with timer() as timing:
            inputs = tokenizer(
                [input_val],
                return_tensors="pt",
            )
        logger.debug(f"Tokenized input in {timing.execution_time_ns / 1e6:.2f}ms")
        
        logger.debug("Performing inference...")
        logger.debug(
            "With config: "
            f"top_k={config.top_k}, "
            f"top_p={config.top_p}, "
            f"temperature={config.temperature}, "
            f"repetition_penalty={config.repetition_penalty}"
        )
        with timer() as timing:
            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=True,
                    num_return_sequences=1,
                    max_length=config.max_length,
                    temperature=config.temperature,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                )
        logger.debug(f"Inferred in {timing.execution_time_ns / 1e6:.2f}ms")

        logger.debug("Decoding outputs...")
        with timer() as timing:
            generated_sequences = []
            for generated_sequence in output_sequences:
                generated_sequence = generated_sequence.tolist()
                # NOTE: the generated sequence includes the input prompt
                text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
                # OPTIONAL: strip the input prompt:
                # decoded_prompt = tokenizer.decode(inputs['input_ids'][0], clean_up_tokenization_spaces=True)
                # text = text[len(decoded_prompt):]
                generated_sequences.append(text)
        logger.debug(f"Decoded outputs in {timing.execution_time_ns / 1e6:.2f}ms")
            
        conn.send(generated_sequences)

    # Clean up resources and exit the child process
    conn.close()


def run_server():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default="gpt2")
    argparser.add_argument("--temperature", type=float, default=0.85)
    argparser.add_argument("--top_k", type=int, default=50)
    argparser.add_argument("--top_p", type=float, default=0.99)
    argparser.add_argument("--max_length", type=int, default=140)
    argparser.add_argument("--repetition_penalty", type=float, default=1.0)
    args = argparser.parse_args()

    config = CompletionConfig(**vars(args))

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=child_process, args=(child_conn, config))
    p.start()

    # Wait for the child process to signal that it's ready to receive inputs
    if not parent_conn.recv():
        raise RuntimeError("Child process failed to load model")

    print("Type a prompt and press 'Enter' to generate a response.")
    print("You can also enter \config to edit the model configuration.")
    print("('ctrl+D' to quit)")
    logger.debug(
        f"top_k={config.top_k}, "
        f"top_p={config.top_p}, "
        f"temperature={config.temperature}, "
        f"repetition_penalty={config.repetition_penalty}",
    )

    try:
        while True:
            input_str = input("Prompt ('ctrl+D' to quit): ")

            # Edit configuration
            if input_str == "\\config":
                cprint(
                    "Current config:\n"
                    f"top_k={config.top_k}, "
                    f"top_p={config.top_p}, "
                    f"temperature={config.temperature}, "
                    f"repetition_penalty={config.repetition_penalty}",
                    "cyan",
                )
                while True:
                    attrname = input(
                        colored(
                            "Enter config attribute to edit (or Enter to leave config): ",
                            "light_cyan"
                        )
                    )
                    if not attrname:
                        break
                    if attrname not in CompletionConfig.__dataclass_fields__:
                        print("Invalid attribute name.")
                        continue
                    attrval = input(
                        colored(f"Enter new value for {attrname}: ", "light_cyan")
                    )
                    setattr(config, attrname, type(getattr(config, attrname))(attrval))
                cprint(
                    "New config:\n"
                    f"top_k={config.top_k}, "
                    f"top_p={config.top_p}, "
                    f"temperature={config.temperature}, "
                    f"repetition_penalty={config.repetition_penalty}",
                    "cyan",
                )
                parent_conn.send(config)
                continue

            parent_conn.send(input_str)

            output = parent_conn.recv()
            print(output[0])
    except (EOFError, KeyboardInterrupt):
        # tell child to close their conn and finish
        print()
        parent_conn.send(None)

    # Clean up resources and wait for the child process to terminate before exiting
    logger.debug("Waiting for child process to terminate...")
    p.join()
    logger.debug("Child process terminated.")


if __name__ == '__main__':
    run_server()
