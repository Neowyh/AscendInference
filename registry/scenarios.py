from dataclasses import dataclass, field
from typing import Any, Dict

from evaluations.routes import RouteType
from evaluations.tiers import InputTier


def _require_field(data, field_name):
    if field_name not in data:
        raise ValueError("Missing required field: %s" % field_name)
    value = data[field_name]
    if value is None or value == "":
        raise ValueError("Missing required field: %s" % field_name)
    return value


@dataclass
class ScenarioDefinition:
    name: str
    model_name: str = ""
    input_tier: InputTier = InputTier.TIER_720P
    route_type: RouteType = RouteType.TILED_ROUTE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_tier = InputTier.from_value(self.input_tier)
        self.route_type = RouteType.from_value(self.route_type)

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=_require_field(data, "name"),
            model_name=_require_field(data, "model_name"),
            input_tier=_require_field(data, "input_tier"),
            route_type=_require_field(data, "route_type"),
            metadata=data.get("metadata", {}),
        )
