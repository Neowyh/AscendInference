from dataclasses import dataclass, field
from typing import Dict

from registry.devices import DeviceProfile
from registry.models import ModelAsset
from registry.scenarios import ScenarioDefinition


@dataclass
class Registry:
    models: Dict[str, ModelAsset] = field(default_factory=dict)
    devices: Dict[str, DeviceProfile] = field(default_factory=dict)
    scenarios: Dict[str, ScenarioDefinition] = field(default_factory=dict)

    def register_model(self, model):
        if model.name in self.models:
            raise ValueError("Duplicate model registration: %s" % model.name)
        self.models[model.name] = model
        return model

    def get_model(self, name):
        return self.models.get(name)

    def register_device(self, device):
        if device.name in self.devices:
            raise ValueError("Duplicate device registration: %s" % device.name)
        self.devices[device.name] = device
        return device

    def get_device(self, name):
        return self.devices.get(name)

    def register_scenario(self, scenario):
        if scenario.name in self.scenarios:
            raise ValueError("Duplicate scenario registration: %s" % scenario.name)
        self.scenarios[scenario.name] = scenario
        return scenario

    def get_scenario(self, name):
        return self.scenarios.get(name)

    @classmethod
    def from_dict(cls, data):
        registry = cls()

        for model_data in data.get("models", []):
            registry.register_model(ModelAsset.from_dict(model_data))

        for device_data in data.get("devices", []):
            registry.register_device(DeviceProfile(**device_data))

        for scenario_data in data.get("scenarios", []):
            registry.register_scenario(ScenarioDefinition(**scenario_data))

        return registry


def load_registry(data=None):
    return Registry.from_dict(data or {})
