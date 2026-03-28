import pytest

from config import SUPPORTED_RESOLUTIONS
from evaluations.routes import RouteType, REMOTE_SENSING_ROUTES
from evaluations.tiers import InputTier, STANDARD_INPUT_TIERS
from registry.devices import DeviceProfile
from registry.loader import Registry, load_registry
from registry.models import InputSpec, ModelAsset
from registry.scenarios import ScenarioDefinition


def test_standard_input_tiers_are_fixed():
    assert STANDARD_INPUT_TIERS == ("720p", "1080p", "4K")
    assert [tier.value for tier in InputTier] == ["720p", "1080p", "4K"]


def test_input_tier_runtime_resolution_mapping_is_explicit():
    assert InputTier.TIER_720P.runtime_resolution == "640x640"
    assert InputTier.TIER_1080P.runtime_resolution == "1k2k"
    assert InputTier.TIER_4K.runtime_resolution == "4k6k"
    assert InputTier.TIER_4K.runtime_resolution in SUPPORTED_RESOLUTIONS


def test_remote_sensing_routes_are_fixed():
    assert REMOTE_SENSING_ROUTES == ("tiled_route", "large_input_route")
    assert [route.value for route in RouteType] == ["tiled_route", "large_input_route"]


def test_model_asset_preserves_input_specs_and_routes():
    asset = ModelAsset(
        name="rs-yolo",
        input_specs=(
            InputSpec(tier=InputTier.TIER_720P, width=1280, height=720),
            InputSpec(tier=InputTier.TIER_4K, width=4096, height=4096),
        ),
        supported_routes=(RouteType.TILED_ROUTE, RouteType.LARGE_INPUT_ROUTE),
    )

    assert asset.name == "rs-yolo"
    assert asset.get_input_spec(InputTier.TIER_720P).width == 1280
    assert asset.get_input_spec("4K").height == 4096
    assert asset.supports_route(RouteType.TILED_ROUTE) is True
    assert asset.supports_route("large_input_route") is True


def test_registry_loader_builds_model_catalog():
    registry = load_registry(
        {
            "models": [
                {
                    "name": "rs-yolo",
                    "input_specs": [
                        {"tier": "720p", "width": 1280, "height": 720},
                        {"tier": "1080p", "width": 1920, "height": 1080},
                    ],
                    "supported_routes": ["tiled_route", "large_input_route"],
                }
            ]
        }
    )

    model = registry.get_model("rs-yolo")

    assert model is not None
    assert model.get_input_spec("1080p").width == 1920
    assert model.supports_route(RouteType.LARGE_INPUT_ROUTE) is True


@pytest.mark.parametrize(
    "payload",
    [
        {"width": 1280, "height": 720},
        {"tier": "720p", "height": 720},
        {"tier": "720p", "width": 1280},
    ],
)
def test_input_spec_from_dict_requires_required_fields(payload):
    with pytest.raises(ValueError):
        InputSpec.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "input_specs": [{"tier": "720p", "width": 1280, "height": 720}],
            "supported_routes": ["tiled_route"],
        },
        {
            "name": "rs-yolo",
            "supported_routes": ["tiled_route"],
        },
        {
            "name": "rs-yolo",
            "input_specs": [{"tier": "720p", "width": 1280, "height": 720}],
        },
    ],
)
def test_model_asset_from_dict_requires_required_fields(payload):
    with pytest.raises(ValueError):
        ModelAsset.from_dict(payload)


@pytest.mark.parametrize(
    "register_name, item_factory",
    [
        (
            "register_model",
            lambda: ModelAsset(
                name="duplicate",
                input_specs=(InputSpec(tier=InputTier.TIER_720P, width=1280, height=720),),
                supported_routes=(RouteType.TILED_ROUTE,),
            ),
        ),
        (
            "register_device",
            lambda: DeviceProfile(
                name="duplicate",
                supported_tiers=(InputTier.TIER_720P,),
                supported_routes=(RouteType.TILED_ROUTE,),
            ),
        ),
        (
            "register_scenario",
            lambda: ScenarioDefinition(
                name="duplicate",
                model_name="duplicate",
                input_tier=InputTier.TIER_720P,
                route_type=RouteType.TILED_ROUTE,
            ),
        ),
    ],
)
def test_registry_rejects_duplicate_registration(register_name, item_factory):
    registry = Registry()
    register = getattr(registry, register_name)

    register(item_factory())

    with pytest.raises(ValueError):
        register(item_factory())


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "name": "",
            "supported_tiers": (InputTier.TIER_720P,),
            "supported_routes": (RouteType.TILED_ROUTE,),
        },
        {
            "name": "edge-device",
            "supported_tiers": [],
            "supported_routes": (RouteType.TILED_ROUTE,),
        },
        {
            "name": "edge-device",
            "supported_tiers": (InputTier.TIER_720P,),
            "supported_routes": [],
        },
        {
            "name": "edge-device",
            "supported_tiers": ("",),
            "supported_routes": (RouteType.TILED_ROUTE,),
        },
    ],
)
def test_device_profile_direct_construction_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DeviceProfile(**kwargs)


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "edge-device"},
        {"supported_tiers": ["720p"]},
        {"supported_routes": ["tiled_route"]},
    ],
)
def test_device_profile_from_dict_requires_required_fields(payload):
    with pytest.raises(ValueError):
        DeviceProfile.from_dict(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"model_name": "rs-yolo"},
        {"name": "edge-scenario"},
        {"route_type": "tiled_route"},
    ],
)
def test_scenario_definition_from_dict_requires_required_fields(payload):
    with pytest.raises(ValueError):
        ScenarioDefinition.from_dict(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "name": "",
            "model_name": "rs-yolo",
            "input_tier": InputTier.TIER_720P,
            "route_type": RouteType.TILED_ROUTE,
        },
        {
            "name": "edge-scenario",
            "model_name": "",
            "input_tier": InputTier.TIER_720P,
            "route_type": RouteType.TILED_ROUTE,
        },
        {
            "name": "edge-scenario",
            "model_name": "rs-yolo",
            "input_tier": "",
            "route_type": RouteType.TILED_ROUTE,
        },
        {
            "name": "edge-scenario",
            "model_name": "rs-yolo",
            "input_tier": InputTier.TIER_720P,
            "route_type": "",
        },
    ],
)
def test_scenario_definition_direct_construction_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ScenarioDefinition(**kwargs)


def test_registry_from_dict_rejects_invalid_device_payload():
    with pytest.raises(ValueError):
        Registry.from_dict(
            {
                "devices": [
                    {
                        "name": "edge-device",
                        "supported_routes": ["tiled_route"],
                    }
                ]
            }
        )


def test_registry_from_dict_rejects_invalid_scenario_payload():
    with pytest.raises(ValueError):
        Registry.from_dict(
            {
                "scenarios": [
                    {
                        "name": "edge-scenario",
                        "route_type": "tiled_route",
                    }
                ]
            }
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
