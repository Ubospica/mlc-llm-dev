# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
from typing import Callable, List, Optional

import numpy as np

from mlc_chat.serve import (
    Engine,
    GenerationConfig,
    KVCacheConfig,
    Request,
    RequestStreamOutput,
    data,
)
from mlc_chat.serve.engine import ModelInfo

prompts = [
    "Generate a JSON string containing 20 objects. Directly return the json string:",
]


def create_requests(
    num_requests: int,
    stop_token_id: Optional[int] = None,
    temperature: float = 0.8,
    repetition_penalty: float = 1.0,
    max_tokens_low: int = 256,
    max_tokens_high: int = 257,
    json_mode: bool = False,
    output_grammar: Optional[str] = None,
) -> List[Request]:
    assert num_requests >= 0 and num_requests <= len(prompts)

    stop_token_ids = [stop_token_id] if stop_token_id is not None else []
    requests = []
    for req_id, prompt in zip(range(num_requests), prompts):
        max_tokens = np.random.randint(max_tokens_low, max_tokens_high)
        requests.append(
            Request(
                request_id=str(req_id),
                inputs=data.TextData(prompt),
                generation_config=GenerationConfig(
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens,
                    stop_token_ids=stop_token_ids,
                    json_mode=json_mode,
                    output_grammar=output_grammar,
                ),
            )
        )
    return requests


def test_engine_basic():
    """Test engine **without continuous batching**.

    - Add all requests to the engine altogether in the beginning.
    - All requests have the same max_tokens. This means all requests
    will end together.
    - Engine keeps running `step` for estimated number of steps (number of
    requests + max_tokens - 1). Then check the output of each request.
    """

    # Initialize model loading info and KV cache config
    model = ModelInfo(
        "dist/Llama-2-7b-chat-hf-q4f16_1-MLC",
        model_lib_path="dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so",
    )
    kv_cache_config = KVCacheConfig(page_size=16)

    # Hyperparameters for tests (you can try different combinations).
    num_requests = 1  # [4, 8, 10]
    temperature = 0.9  # [0, 0.8, 0.9, 1.0, 1.1]
    repetition_penalty = 1.0  # [1.0, 1.01]
    max_tokens: int = 1024  # [32, 128, 256]
    np.random.seed(0)

    # Output list
    outputs = [[] for _ in range(num_requests)]

    # Define the callback function for request generation results
    def fcallback(delta_outputs: List[RequestStreamOutput]):
        for delta_output in delta_outputs:
            request_id, delta_tokens, _ = delta_output.unpack()
            outputs[int(request_id)] += delta_tokens.token_ids

    # Create engine
    engine = Engine(model, kv_cache_config, request_stream_callback=fcallback)

    # Create requests
    requests = create_requests(
        num_requests,
        stop_token_id=2,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        max_tokens_low=max_tokens,
        max_tokens_high=max_tokens + 1,
        json_mode=True,
    )

    # Add all requests to engine
    for request in requests:
        engine.add_request(request)

    num_steps = num_requests + max_tokens - 1
    # Run steps
    for step in range(num_steps):
        engine.step()

    for req_id, output in enumerate(outputs):
        print(f"Prompt {req_id}: {requests[req_id].inputs[0]}")
        print(f"Output {req_id}:{engine.tokenizer.decode(output)}\n")


if __name__ == "__main__":
    test_engine_basic()
