import json
from pathlib import Path

import pytest

from config import Config
from config.strategy_config import EvaluationConfig
from config.validator import ConfigValidator
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


def test_evaluation_templates_exist_and_expose_evaluation_block():
    base_dir = Path(__file__).resolve().parents[1] / "config" / "evaluation"

    standard_path = base_dir / "default_standard_eval.json"
    remote_sensing_path = base_dir / "default_remote_sensing_eval.json"

    assert standard_path.exists()
    assert remote_sensing_path.exists()

    standard_data = json.loads(standard_path.read_text(encoding="utf-8"))
    remote_sensing_data = json.loads(remote_sensing_path.read_text(encoding="utf-8"))

    assert standard_data["evaluation"]["input_tier"] == "720p"
    assert standard_data["evaluation"]["route_type"] == "tiled_route"
    assert remote_sensing_data["evaluation"]["input_tier"] == "4K"
    assert remote_sensing_data["evaluation"]["route_type"] == "large_input_route"


def test_large_input_route_with_supported_combinations_is_valid(monkeypatch):
    config = Config(
        model_path="models/yolov8s.om",
        resolution="1k2k",
        evaluation=EvaluationConfig(
            input_tier="1080p",
            route_type="large_input_route",
            report_format="json",
            archive_enabled=True,
        ),
    )

    monkeypatch.setattr("config.validator.os.path.exists", lambda _: True)

    result = ConfigValidator.validate(config)

    assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
