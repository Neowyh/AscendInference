from enum import Enum


REMOTE_SENSING_ROUTES = ("tiled_route", "large_input_route")


class RouteType(str, Enum):
    TILED_ROUTE = "tiled_route"
    LARGE_INPUT_ROUTE = "large_input_route"

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        for route in cls:
            if route.value == value:
                return route
        raise ValueError("Unsupported route type: %s" % value)
