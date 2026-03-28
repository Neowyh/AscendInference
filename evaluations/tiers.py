from enum import Enum


STANDARD_INPUT_TIERS = ("720p", "1080p", "4K")


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
