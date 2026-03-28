from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from evaluations.routes import RouteType
from evaluations.tiers import InputTier
from registry.models import InputSpec


@dataclass
class EvaluationTask:
    model_name: str
    input_tier: InputTier
    route_type: RouteType
    input_spec: Optional[InputSpec] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.input_tier = InputTier.from_value(self.input_tier)
        self.route_type = RouteType.from_value(self.route_type)
        if self.input_spec is not None and not isinstance(self.input_spec, InputSpec):
            self.input_spec = InputSpec.from_dict(self.input_spec)

    @classmethod
    def from_model_asset(cls, asset, input_tier, route_type, metadata=None):
        input_tier = InputTier.from_value(input_tier)
        route_type = RouteType.from_value(route_type)

        if not asset.supports_route(route_type):
            raise ValueError("Model does not support route: %s" % route_type.value)

        input_spec = asset.get_input_spec(input_tier)
        return cls(
            model_name=asset.name,
            input_tier=input_tier,
            route_type=route_type,
            input_spec=input_spec,
            metadata=metadata or {},
        )

    @classmethod
    def from_scenario(cls, scenario, asset, metadata=None):
        if scenario.model_name and scenario.model_name != asset.name:
            raise ValueError("Scenario model does not match asset")

        return cls.from_model_asset(
            asset,
            input_tier=scenario.input_tier,
            route_type=scenario.route_type,
            metadata=metadata or scenario.metadata,
        )
