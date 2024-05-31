# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
import json
from typing import Dict, List, Literal

import pytest
from pydantic import BaseModel

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.protocol.openai_api_protocol import RequestResponseFormat as ResponseFormat
from mlc_llm.serve import AsyncMLCEngine
from mlc_llm.serve.sync_engine import SyncMLCEngine
from mlc_llm.testing import require_test_model
from mlc_llm.protocol.openai_api_protocol import ChatCompletionResponse
from mlc_llm.serve import AsyncMLCEngine, EngineConfig, GenerationConfig, MLCEngine
from mlc_llm.serve.config import ResponseFormat

prompts_list = [
    "Generate a JSON string containing 20 objects:",
    "Generate a JSON containing a non-empty list:",
    "Generate a JSON with 5 elements:",
]
# use a model finetuned for chat or json output
model_path = "dist/Llama-2-7b-chat-hf-q4f16_1-MLC"
model_lib_path = "dist/libs/Llama-2-7b-chat-hf-q4f16_1-cuda.so"


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_batch_generation_with_grammar(model: str):
    # Create engine
    engine = MLCEngine(
        model=model,
        mode="server",
        max_total_sequence_length=4096,
        engine_config=EngineConfig(debug_disable_jump_forward=True),
    )

    responses: List[ChatCompletionResponse] = []
    temperature = 1
    max_tokens = 512

    # non-json output
    for p in prompts_list:
        responses.append(
            engine.chat.completions.create(
                messages=[{"role": "user", "content": p}],
                model=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "text"},
            )
        )

    # json output
    for p in prompts_list:
        responses.append(
            engine.chat.completions.create(
                messages=[{"role": "user", "content": p}],
                model=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "json_object"},
            )
        )

    # Generate output.
    for req_id, response in enumerate(responses):
        prompt = prompts_list[req_id % 3]
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    engine.terminate()


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
def test_batch_generation_with_schema(model: str):
    # Create engine
    engine = MLCEngine(
        model=model,
        mode="server",
        max_total_sequence_length=4096,
        engine_config=EngineConfig(debug_disable_jump_forward=True),
    )

    prompt = (
        "Generate a json containing three fields: an integer field named size, a "
        "boolean field named is_accepted, and a float field named num:"
    )
    repeat_cnt = 3

    responses: List[ChatCompletionResponse] = []
    temperature = 1
    max_tokens = 512

    # without schema
    for _ in range(repeat_cnt):
        responses.append(
            engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "text"},
            )
        )

    class Schema(BaseModel):
        size: int
        is_accepted: bool
        num: float

    schema_str = json.dumps(Schema.model_json_schema())

    # with schema
    for _ in range(repeat_cnt):
        responses.append(
            engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "json_object", "schema": schema_str},
            )
        )

    for req_id, response in enumerate(responses):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    engine.terminate()


def test_batch_generation_jump_forward():
    # Create engine
    engine = MLCEngine(
        model=model_path,
        model_lib_path=model_lib_path,
        mode="server",
        max_total_sequence_length=4096,
        engine_config=EngineConfig(debug_disable_jump_forward=False),
    )

    prompt = (
        'Generate a json string that includes seven fields: "size" which is an integer, '
        '"is_accepted" which is a boolean, "num" which is a floating-point number, '
        '"is_correct" which must be true, "message" which is a string that can only be '
        '"aaaaaaaaa" or "bbbbbbbbb" or "ccccccccc", "array_field" which is an array of '
        'strings, and "object_field" which is an object with keys as strings and values'
        " as integers:"
    )
    repeat_cnt = 20

    responses: List[ChatCompletionResponse] = []
    temperature = 1
    max_tokens = 512

    class Schema(BaseModel):
        size: int
        is_accepted: bool
        num: float
        is_correct: Literal[True]
        message: Literal["aaaaaaaaa", "bbbbbbbbb", "ccccccccc"]
        array_field: List[str]
        object_field: Dict[str, int]

    schema_str = json.dumps(Schema.model_json_schema())

    for _ in range(repeat_cnt):
        responses.append(
            engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                response_format={"type": "json_object", "schema": schema_str},
            )
        )

    for req_id, response in enumerate(responses):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    engine.terminate()


class Schema(BaseModel):
    size: int
    is_accepted: bool
    num: float
    is_correct: Literal[True]
    message: Literal["aaaaaaaaa", "bbbbbbbbb", "ccccccccc"]
    array_field: List[str]
    object_field: Dict[str, int]


schema_str = json.dumps(Schema.model_json_schema())
print(schema_str.replace('"', '\\"'))
exit()

test_batch_generation_jump_forward()
exit()


@require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
async def run_async_engine(model: str):
    # Create engine
    async_engine = AsyncMLCEngine(model=model, mode="server")

    prompts = prompts_list * 20

    max_tokens = 256
    temperature = 1
    max_tokens = 512
    generation_config = GenerationConfig(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=[2],
        response_format=ResponseFormat(type="json_object"),
    )

    output_texts: List[List[str]] = [
        ["" for _ in range(generation_config.n)] for _ in range(len(prompts))
    ]

    async def generate_task(
        async_engine: AsyncMLCEngine,
        prompt: str,
        generation_cfg: GenerationConfig,
        request_id: str,
    ):
        print(f"Start generation task for request {request_id}")
        rid = int(request_id)
        async for delta_outputs in async_engine._generate(
            prompt, generation_cfg, request_id=request_id
        ):
            assert len(delta_outputs) == generation_cfg.n
            for i, delta_output in enumerate(delta_outputs):
                output_texts[rid][i] += delta_output.delta_text

    tasks = [
        asyncio.create_task(
            generate_task(async_engine, prompts[i], generation_config, request_id=str(i))
        )
        for i in range(len(prompts))
    ]

    await asyncio.gather(*tasks)

    # Print output.
    print("All finished")
    for req_id, outputs in enumerate(output_texts):
        print(f"Prompt {req_id}: {prompts[req_id]}")
        if len(outputs) == 1:
            print(f"Output {req_id}:{outputs[0]}\n")
        else:
            for i, output in enumerate(outputs):
                print(f"Output {req_id}({i}):{output}\n")

    async_engine.terminate()


def test_async_engine():
    asyncio.run(run_async_engine())


def test_generation_config_error():
    with pytest.raises(ValueError):
        GenerationConfig(
            temperature=1.0,
            repetition_penalty=1.0,
            max_tokens=128,
            stop_token_ids=[2],
            response_format=ResponseFormat(type="text", schema="{}"),
        )


if __name__ == "__main__":
    test_batch_generation_with_grammar()
    test_async_engine()
    test_generation_config_error()
