from splifft import PATH_REGISTRY_DEFAULT
from splifft.config import Registry


def test_registry_json_validates_with_pydantic() -> None:
    registry = Registry.from_file(PATH_REGISTRY_DEFAULT)
    assert isinstance(registry, Registry)
