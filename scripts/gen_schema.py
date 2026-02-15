import json
from typing import Any

from pydantic import TypeAdapter

from splifft import DIR_DATA
from splifft.config import Config, Registry


def write_schema(file: str, obj: dict[str, Any]) -> None:
    (DIR_DATA / file).write_text(json.dumps(obj, indent=2))


if __name__ == "__main__":
    write_schema("config.schema.json", Config.model_json_schema())
    write_schema("registry.schema.json", TypeAdapter(Registry).json_schema())
