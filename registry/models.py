from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from evaluations.routes import RouteType
from evaluations.tiers import InputTier


def _coerce_input_tier(value):
    return InputTier.from_value(value)


def _coerce_route_type(value):
    return RouteType.from_value(value)


@dataclass
class InputSpec:
    tier: InputTier
    width: int
    height: int
    channels: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.tier = _coerce_input_tier(self.tier)

    @classmethod
    def from_dict(cls, data):
        return cls(
            tier=data.get("tier"),
            width=data.get("width", 0),
            height=data.get("height", 0),
            channels=data.get("channels", 3),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self):
        return {
            "tier": self.tier.value,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "metadata": dict(self.metadata),
        }


@dataclass
class ModelAsset:
    name: str
    input_specs: Tuple[InputSpec, ...] = field(default_factory=tuple)
    supported_routes: Tuple[RouteType, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_specs = tuple(
            spec if isinstance(spec, InputSpec) else InputSpec.from_dict(spec)
            for spec in self.input_specs
        )
        self.supported_routes = tuple(
            route if isinstance(route, RouteType) else _coerce_route_type(route)
            for route in self.supported_routes
        )

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data.get("name", ""),
            input_specs=tuple(data.get("input_specs", ())),
            supported_routes=tuple(data.get("supported_routes", ())),
            metadata=data.get("metadata", {}),
        )

    def get_input_spec(self, tier):
        input_tier = _coerce_input_tier(tier)
        for spec in self.input_specs:
            if spec.tier == input_tier:
                return spec
        raise KeyError("Unsupported input tier: %s" % input_tier.value)

    def supports_route(self, route):
        return _coerce_route_type(route) in self.supported_routes
