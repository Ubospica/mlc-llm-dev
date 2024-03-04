import json
from enum import Enum
from typing import Union

from ebnf import grammar, parse
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict

from mlc_chat.serve.json_schema_converter import json_schema_to_bnf


def test_simple():
    class FooBar(BaseModel):
        count: int
        size: Union[float, None] = None

    class Gender(str, Enum):
        male = "male"
        female = "female"
        other = "other"
        not_given = "not_given"

    class MainModel(BaseModel):
        """
        This is the description of the main model
        """

        model_config = ConfigDict(title="Main")

        foo_bar: FooBar
        gender: Union[Gender, None] = None
        snap: int = Field(
            42,
            title="The Snap",
            description="this is the value of snap",
            gt=30,
            lt=50,
        )

    main_model_schema = MainModel.model_json_schema()
    main_model_schema_str = json.dumps(main_model_schema, indent=2)
    print(main_model_schema_str)
    bnf_grammar = json_schema_to_bnf(main_model_schema_str)

    object_str = '{"foo_bar": {"count": 42, "size": 3.1}, "gender": "male", "snap": 42}'
    assert parse(object_str, *grammar(bnf_grammar))


if __name__ == "__main__":
    test_simple()
