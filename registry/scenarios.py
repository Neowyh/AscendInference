from dataclasses import dataclass, field
from typing import Any, Dict

from evaluations.routes import RouteType
from evaluations.tiers import InputTier


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
