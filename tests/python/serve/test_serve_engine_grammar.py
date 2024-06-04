# pylint: disable=chained-comparison,line-too-long,missing-docstring,
# pylint: disable=too-many-arguments,too-many-locals,unused-argument,unused-variable
import asyncio
import json
import random
from typing import Dict, List, Literal

import pytest
from pydantic import BaseModel

from mlc_llm.protocol.generation_config import GenerationConfig
from mlc_llm.protocol.openai_api_protocol import (
    ChatCompletionResponse,
    RequestResponseFormat,
)
from mlc_llm.serve import AsyncMLCEngine, EngineConfig, MLCEngine
from mlc_llm.testing import require_test_model

# random.seed(122)


# @require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
@require_test_model("Meta-Llama-3-8B-Instruct-q4f16_1-MLC")
def test_batch_generation_with_grammar(model: str):
    # Engine
    engine = MLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(debug_disable_jump_forward=True),
    )

    # Inputs
    system_prompt = "You are a helpful assistant. Always respond only with json."
    prompts_list = [
        "Generate a JSON string containing 20 objects:",
        "Generate a JSON containing a non-empty list:",
        "Generate a JSON with 5 elements:",
        "Generate a JSON with a number list, counting from 1 to 20:",
    ]

    repeat = 3
    top_p = 1
    temperature = 1

    # non-json output
    responses_text: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        for p in prompts_list:
            responses_text.append(
                engine.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ],
                    response_format={"type": "text"},
                    top_p=top_p,
                    temperature=temperature,
                    seed=random.randint(0, 1 << 30),
                )
            )

    print("Text output")
    for req_id, response in enumerate(responses_text):
        prompt = prompts_list[req_id % len(prompts_list)]
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    # json output
    responses_json: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        for p in prompts_list:
            responses_json.append(
                engine.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": p},
                    ],
                    response_format={"type": "json_object"},
                    top_p=top_p,
                    temperature=temperature,
                    seed=random.randint(0, 1 << 30),
                )
            )

    print("JSON output")
    for req_id, response in enumerate(responses_json):
        prompt = prompts_list[req_id % len(prompts_list)]
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")
        json.loads(output)

    engine.terminate()


# test_batch_generation_with_grammar()
# exit()


# @require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
@require_test_model("Meta-Llama-3-8B-Instruct-q4f16_1-MLC")
def test_batch_generation_with_schema(model: str):
    # Create engine
    engine = MLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(debug_disable_jump_forward=False),
    )

    class Product(BaseModel):
        product_id: int
        is_available: bool
        price: float
        is_featured: Literal[True]
        category: Literal["Electronics", "Clothing", "Food"]
        tags: List[str]
        stock: Dict[str, int]

    schema_str = json.dumps(Product.model_json_schema())

    system_prompt = (
        "You are a helpful assistant. Always respond only with JSON based on the "
        f"following JSON schema: {schema_str}."
    )
    prompt = "Generate a JSON that describes the product according to the given JSON schema."

    repeat = 20
    top_p = 1
    temperature = 1

    # non-json output
    responses_text: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        responses_text.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "text"},
                top_p=top_p,
                temperature=temperature,
                seed=random.randint(0, 1 << 30),
            )
        )

    print("Text output")
    for req_id, response in enumerate(responses_text):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    # json output without schema
    responses_json: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        responses_json.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                top_p=top_p,
                temperature=temperature,
                seed=random.randint(0, 1 << 30),
            )
        )

    print("JSON output")
    for req_id, response in enumerate(responses_json):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    # json output with schema
    responses_schema: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        responses_schema.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object", "schema": schema_str},
                top_p=top_p,
                temperature=temperature,
                seed=random.randint(0, 1 << 30),
            )
        )

    print("JSON Schema output")
    for req_id, response in enumerate(responses_schema):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    engine.terminate()


# @require_test_model("Llama-2-7b-chat-hf-q4f16_1-MLC")
@require_test_model("Meta-Llama-3-8B-Instruct-q4f16_1-MLC")
def test_batch_generation_jump_forward(model):
    # Create engine
    engine = MLCEngine(
        model=model,
        mode="server",
        engine_config=EngineConfig(debug_disable_jump_forward=False),
    )

    class Product(BaseModel):
        product_id: int
        is_available: bool
        price: float
        is_featured: Literal[True]
        category: Literal["Electronics", "Clothing", "Food"]
        tags: List[str]
        stock: Dict[str, int]

    schema_str = json.dumps(Product.model_json_schema())

    system_prompt = (
        "You are a helpful assistant. Always respond only with JSON based on the "
        f"following JSON schema: {schema_str}."
    )
    prompt = "Generate a JSON that describes the product according to the given JSON schema."

    repeat = 1
    top_p = 0.9
    temperature = 0.6

    # # json output without schema
    # responses_json: List[ChatCompletionResponse] = []
    # for _ in range(repeat):
    #     responses_json.append(
    #         engine.chat.completions.create(
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": prompt},
    #             ],
    #             response_format={"type": "json_object"},
    #             top_p=top_p,
    #             temperature=temperature,
    #             seed=random.randint(0, 1 << 30),
    #         )
    #     )

    # print("JSON output")
    # for req_id, response in enumerate(responses_json):
    #     output = response.choices[0].message.content
    #     print(f"Prompt {req_id}: {prompt}")
    #     print(f"Output {req_id}: {output}\n")

    # json output with schema
    responses_schema: List[ChatCompletionResponse] = []
    for _ in range(repeat):
        print("gen start\n")
        responses_schema.append(
            engine.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object", "schema": schema_str},
                top_p=top_p,
                temperature=temperature,
                seed=random.randint(0, 1 << 30),
                logprobs=True,
                top_logprobs=5,
            )
        )
        print("\ngen end\n")

    print("JSON Schema output")
    for req_id, response in enumerate(responses_schema):
        output = response.choices[0].message.content
        print(f"Prompt {req_id}: {prompt}")
        print(f"Output {req_id}: {output}\n")

    engine.terminate()


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
        response_format=RequestResponseFormat(type="json_object"),
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
            response_format=RequestResponseFormat(type="text", schema="{}"),
        )


if __name__ == "__main__":
    test_batch_generation_with_grammar()
    test_async_engine()
    test_generation_config_error()
