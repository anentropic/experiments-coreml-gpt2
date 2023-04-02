import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from loader import load_coreml
from utils import timer


logger.configure(handlers=[
    {
        "sink": sys.stderr,
        "format": "<light-black>{message}</light-black>",
        # "enqueue": True,
    }
])

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_RESULT_KEYS = {
    0: "negative",
    1: "positive",
}


def child_process(conn, model_path):
    try:
        mlmodel, tokenizer = load_coreml(str(PROJECT_ROOT / model_path))
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
        input_str = conn.recv()

        # If the input is None, exit the loop and terminate the child process
        if input_str is None:
            break

        logger.debug("Tokenizing input...")
        with timer() as timing:
            inputs = tokenizer(
                [input_str],
                return_tensors="np",
                max_length=64,
                padding="max_length",
            )
        logger.debug(f"Tokenized input in {timing.execution_time_ns / 1e6:.2f}ms")
        
        logger.debug("Performing inference...")
        with timer() as timing:
            result = mlmodel.predict({
                "input_ids": inputs["input_ids"][0].astype(np.int32),
                "position_ids": inputs["attention_mask"][0].astype(np.int32),
            })
        logger.debug(f"Inferred in {timing.execution_time_ns / 1e6:.2f}ms")

        logger.debug(result)

        output_indices = np.argmax(result['output_logits'], axis=-1)  # which axis?

        conn.send(result.tolist())

    # Clean up resources and exit the child process
    conn.close()


def run_server():
    model_path = sys.argv[1]

    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=child_process, args=(child_conn, model_path))
    p.start()

    # Wait for the child process to signal that it's ready to receive inputs
    if not parent_conn.recv():
        raise RuntimeError("Child process failed to load model")

    try:
        while True:
            input_str = input("Prompt (or 'ctrl+D' to quit): ")
            parent_conn.send(input_str)

            output = parent_conn.recv()
            print(output)
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
