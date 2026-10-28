"""Bounded YAML syntax decoding, independent of catalog lookup or model execution."""

import yaml
from pydantic import ValidationError

from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import MAX_NESTING, MAX_YAML_BYTES, Recipe


class RecipeLoader(yaml.SafeLoader):
    """Safe YAML construction that never silently overwrites a mapping key."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[str, object]:
        """Construct string keys once, refusing duplicate keys and YAML merge syntax."""
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key == "<<" or key in result:
                raise ArchitectureError("invalid_yaml", "duplicate, merged or non-string YAML key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def decode_yaml(text: str) -> object:
    """Decode bounded YAML with no explicit tags, anchors, aliases or deep structures."""
    if len(text.encode("utf-8")) > MAX_YAML_BYTES:
        raise ArchitectureError("yaml_byte_limit", "recipe exceeds YAML byte budget")
    depth = 0
    try:
        for event in yaml.parse(text):
            if isinstance(event, yaml.AliasEvent) or getattr(event, "anchor", None):
                raise ArchitectureError("invalid_yaml", "YAML anchors and aliases are unsupported")
            if getattr(event, "tag", None) is not None:
                raise ArchitectureError("invalid_yaml", "explicit YAML tags are unsupported")
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > MAX_NESTING * 4:
                    raise ArchitectureError("yaml_nesting_limit", "YAML nesting exceeds budget")
            if isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
        return yaml.load(text, Loader=RecipeLoader)
    except yaml.YAMLError as error:
        raise ArchitectureError("invalid_yaml", "malformed YAML") from error


def parse_recipe(text: str) -> Recipe:
    """Parse version-one declarations; defer dependency/scope/tensor semantics to resolution."""
    try:
        return Recipe.model_validate(decode_yaml(text))
    except ValidationError as error:
        raise ArchitectureError("invalid_recipe", "recipe declaration is invalid") from error
