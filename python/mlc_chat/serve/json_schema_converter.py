import json


class JSONSchemaToBNF:
    pass


def json_schema_to_bnf(json_schema: str) -> str:
    """Converts a JSON schema to a BNF grammar."""
    json_schema_obj = json.loads(json_schema)
