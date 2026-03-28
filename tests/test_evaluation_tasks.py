import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from commands.model_bench import create_parser, run_benchmark
from config import Config
from config.strategy_config import EvaluationConfig
from config.validator import ConfigValidator
from evaluations.routes import RouteType
from evaluations.tasks import EvaluationTask
from evaluations.tiers import InputTier, STANDARD_INPUT_TIERS
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


def test_model_bench_parser_defaults_standard_input_tiers():
    parser = create_parser()

    args = parser.parse_args(["model.om", "--images", "image.jpg"])

    assert args.input_tiers == list(STANDARD_INPUT_TIERS)


def test_model_bench_parser_rejects_unknown_input_tier():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["model.om", "--images", "image.jpg", "--input-tiers", "6K"])


def test_model_bench_run_preserves_input_tiers_in_scenario_config():
    args = Mock(
        models=["model.om"],
        images=["image.jpg"],
        iterations=10,
        warmup=2,
        output=None,
        format="text",
        device=0,
        backend="pil",
        enable_monitoring=False,
        input_tiers=["720p", "4K"],
    )
    scenario = Mock()
    scenario.run.return_value = [Mock()]
    scenario.generate_report.return_value = "report"

    with patch("commands.model_bench.ModelSelectionScenario", return_value=scenario) as scenario_cls:
        result = run_benchmark(args)

    assert result == 0
    assert scenario_cls.call_args.args[0]["input_tiers"] == ["720p", "4K"]


def test_model_bench_run_preserves_device_and_backend_in_scenario_config():
    args = Mock(
        models=["model.om"],
        images=["image.jpg"],
        iterations=10,
        warmup=2,
        output=None,
        format="text",
        device=3,
        backend="opencv",
        enable_monitoring=False,
        input_tiers=["720p"],
    )
    scenario = Mock()
    scenario.run.return_value = [Mock()]
    scenario.generate_report.return_value = "report"

    with patch("commands.model_bench.ModelSelectionScenario", return_value=scenario) as scenario_cls:
        result = run_benchmark(args)

    assert result == 0
    assert scenario_cls.call_args.args[0]["device_id"] == 3
    assert scenario_cls.call_args.args[0]["backend"] == "opencv"


def test_main_model_bench_parser_passes_explicit_input_tiers(monkeypatch):
    import main as main_module

    captured = {}

    def fake_cmd(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(main_module, "_cmd_model_bench", fake_cmd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "model-bench",
            "model.om",
            "--images",
            "image.jpg",
            "--input-tiers",
            "720p",
            "4K",
        ],
    )

    exit_code = main_module.main()

    assert exit_code == 0
    assert captured["args"].input_tiers == ["720p", "4K"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
