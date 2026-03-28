from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from evaluations.routes import RouteType
from evaluations.tiers import InputTier


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

    def supports_tier(self, tier):
        value = InputTier.from_value(tier)
        return value in self.supported_tiers

    def supports_route(self, route):
        value = RouteType.from_value(route)
        return value in self.supported_routes
