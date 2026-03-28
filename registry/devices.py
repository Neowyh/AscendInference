from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

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
class DeviceProfile:
    name: str
    device_id: str = ""
    supported_tiers: Tuple[InputTier, ...] = field(default_factory=tuple)
    supported_routes: Tuple[RouteType, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.supported_tiers = tuple(
            tier if isinstance(tier, InputTier) else InputTier.from_value(tier)
            for tier in self.supported_tiers
        )
        self.supported_routes = tuple(
            route if isinstance(route, RouteType) else RouteType.from_value(route)
            for route in self.supported_routes
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=_require_field(data, "name"),
            device_id=data.get("device_id", ""),
            supported_tiers=tuple(_require_field(data, "supported_tiers")),
            supported_routes=tuple(_require_field(data, "supported_routes")),
            metadata=data.get("metadata", {}),
        )

    def supports_tier(self, tier):
        value = InputTier.from_value(tier)
        return value in self.supported_tiers

    def supports_route(self, route):
        value = RouteType.from_value(route)
        return value in self.supported_routes
