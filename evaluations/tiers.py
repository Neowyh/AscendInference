from enum import Enum

from config import SUPPORTED_RESOLUTIONS


class InputTier(str, Enum):
    TIER_720P = "720p"
    TIER_1080P = "1080p"
    TIER_4K = "4K"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        for tier in cls:
            if tier.value == value:
                return tier
        raise ValueError("Unsupported input tier: %s" % value)

    @property
    def runtime_resolution(self):
        return PLAN_TIER_TO_RUNTIME_RESOLUTION[self]


STANDARD_INPUT_TIERS = tuple(tier.value for tier in InputTier)


PLAN_TIER_TO_RUNTIME_RESOLUTION = {
    InputTier.TIER_720P: "640x640",
    InputTier.TIER_1080P: "1k2k",
    InputTier.TIER_4K: "4k6k",
}


for runtime_resolution in PLAN_TIER_TO_RUNTIME_RESOLUTION.values():
    if runtime_resolution not in SUPPORTED_RESOLUTIONS:
        raise RuntimeError(
            "Unsupported runtime resolution mapping: %s" % runtime_resolution
        )
