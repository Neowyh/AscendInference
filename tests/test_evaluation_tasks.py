import pytest

from evaluations.routes import RouteType
from evaluations.tasks import EvaluationTask
from evaluations.tiers import InputTier
from registry.models import InputSpec, ModelAsset


def test_evaluation_task_keeps_tier_and_route():
    task = EvaluationTask(
        model_name="rs-yolo",
        input_tier=InputTier.TIER_1080P,
        route_type=RouteType.LARGE_INPUT_ROUTE,
    )

    assert task.model_name == "rs-yolo"
    assert task.input_tier.value == "1080p"
    assert task.route_type.value == "large_input_route"


def test_evaluation_task_can_be_built_from_model_asset():
    asset = ModelAsset(
        name="rs-yolo",
        input_specs=(
            InputSpec(tier=InputTier.TIER_720P, width=1280, height=720),
            InputSpec(tier=InputTier.TIER_4K, width=4096, height=4096),
        ),
        supported_routes=(RouteType.TILED_ROUTE, RouteType.LARGE_INPUT_ROUTE),
    )

    task = EvaluationTask.from_model_asset(
        asset,
        input_tier=InputTier.TIER_4K,
        route_type=RouteType.TILED_ROUTE,
    )

    assert task.model_name == "rs-yolo"
    assert task.input_spec.width == 4096
    assert task.input_spec.height == 4096
    assert task.route_type is RouteType.TILED_ROUTE


def test_evaluation_task_rejects_unsupported_route():
    asset = ModelAsset(
        name="rs-yolo",
        input_specs=(InputSpec(tier=InputTier.TIER_720P, width=1280, height=720),),
        supported_routes=(RouteType.TILED_ROUTE,),
    )

    with pytest.raises(ValueError):
        EvaluationTask.from_model_asset(
            asset,
            input_tier=InputTier.TIER_720P,
            route_type=RouteType.LARGE_INPUT_ROUTE,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
